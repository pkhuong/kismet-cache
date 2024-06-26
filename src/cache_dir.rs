use std::borrow::Cow;
use std::fs::File;
use std::io::Error;
use std::io::ErrorKind;
use std::io::Result;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

use crate::benign_error::is_absent_file_error;
use crate::raw_cache;
use crate::trigger::PeriodicTrigger;

/// Delete temporary files with mtime older than this age.
#[cfg(not(test))]
const MAX_TEMP_FILE_AGE: Duration = Duration::from_secs(3600);

// We want a more eager timeout in tests.
#[cfg(test)]
const MAX_TEMP_FILE_AGE: Duration = Duration::from_secs(2);

/// Attempts to make sure `path` is a directory that exists.  Unlike
/// `std::fs::create_dir_all`, this function is optimised for the case
/// where `path` is already a directory.
fn ensure_directory(path: &Path) -> Result<()> {
    if let Ok(meta) = std::fs::metadata(path) {
        if meta.file_type().is_dir() {
            return Ok(());
        }
    }

    std::fs::create_dir_all(path)
}

/// Deletes any file with mtime older than `MAX_TEMP_FILE_AGE` in
/// `temp_dir`.
///
/// It is not an error if `temp_dir` does not exist.
fn cleanup_temporary_directory(temp_dir: Cow<Path>) -> Result<()> {
    let threshold = match std::time::SystemTime::now().checked_sub(MAX_TEMP_FILE_AGE) {
        Some(time) => time,
        None => return Ok(()),
    };

    let iter = match std::fs::read_dir(&temp_dir) {
        Err(e) if is_absent_file_error(&e) => return Ok(()),
        x => x?,
    };

    let mut temp = temp_dir.into_owned();
    for dirent in iter.flatten() {
        let mut handle = || -> Result<()> {
            let metadata = dirent.metadata()?;
            let mtime = metadata.modified()?;

            if mtime < threshold {
                temp.push(dirent.file_name());
                let ret = std::fs::remove_file(&temp);
                temp.pop();

                ret?;
            }

            Ok(())
        };

        let _ = handle();
    }

    Ok(())
}

/// Returns `name` if it non-empty and does not start with a reserved
/// byte (dot, slash, backslash).
fn validate_file_name(name: &str) -> Result<&str> {
    match name.as_bytes().first() {
        None => Err(Error::new(
            ErrorKind::InvalidInput,
            "kismet cached file name must not be empty",
        )),
        Some(b'.') => Err(Error::new(
            ErrorKind::InvalidInput,
            "kismet cached file name must not start with a dot",
        )),
        Some(b'/') => Err(Error::new(
            ErrorKind::InvalidInput,
            "kismet cached file name must not starts with a forward slash",
        )),
        Some(b'\\') => Err(Error::new(
            ErrorKind::InvalidInput,
            "kismet cached file name must not starts with a backslash",
        )),
        Some(_) => Ok(name),
    }
}

/// The `CacheDir` trait drives the actual management of a single
/// cache directory.
pub(crate) trait CacheDir {
    /// Return the path for the cache directory's temporary
    /// subdirectory.
    fn temp_dir(&self) -> Cow<Path>;
    /// Returns the path for the cache directory.
    fn base_dir(&self) -> Cow<Path>;
    /// Returns the cache directory's trigger object.
    fn trigger(&self) -> &PeriodicTrigger;
    /// Returns the cache's directory capacity (in object count).
    fn capacity(&self) -> usize;

    /// Return the path for the cache directory's temporary
    /// subdirectory, after making sure the directory exists.
    fn ensure_temp_dir(&self) -> Result<Cow<Path>> {
        let ret = self.temp_dir();
        ensure_directory(&ret)?;
        Ok(ret)
    }

    /// Returns a read-only file for `name` in the cache directory if
    /// it exists, or None if there is no such file.
    ///
    /// Implicitly "touches" the cached file `name` if it exists.
    fn get(&self, name: &str) -> Result<Option<File>> {
        let name = validate_file_name(name)?;
        let mut target = self.base_dir().into_owned();
        target.push(name);

        match File::open(&target) {
            Ok(file) => {
                let _ = raw_cache::ensure_file_touched(&file);
                Ok(Some(file))
            }
            Err(e) if is_absent_file_error(&e) => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Deletes old files in this cache directory's subdirectory of
    /// temporary files.
    fn cleanup_temp_directory(&self) -> Result<()> {
        cleanup_temporary_directory(self.temp_dir())
    }

    /// Updates the second chance cache state and deletes temporary
    /// files in the `base_dir` cache directory.
    fn definitely_cleanup(&self, base_dir: PathBuf) -> Result<u64> {
        let ret = match raw_cache::prune(base_dir, self.capacity()) {
            Ok((estimate, _deleted)) => estimate,
            Err(e) if is_absent_file_error(&e) => return Ok(0),
            Err(e) => return Err(e),
        };

        // Delete old temporary files while we're here.
        self.cleanup_temp_directory()?;
        Ok(ret)
    }

    /// If a periodic cleanup is called for, updates the second chance
    /// cache state and deletes temporary files in that cache directory.
    ///
    /// Returns the estimated number of files remaining after cleanup
    /// whenever cleanup was initiated.
    fn maybe_cleanup(&self, base_dir: &Path) -> Result<Option<u64>> {
        if self.trigger().event() {
            Ok(Some(self.definitely_cleanup(base_dir.to_owned())?))
        } else {
            Ok(None)
        }
    }

    /// Updates the second chance cache state and deletes temporary
    /// files in the `base_dir` cache directory.
    ///
    /// Returns the estimated number of files remaining after cleanup.
    fn maintain(&self) -> Result<u64> {
        self.definitely_cleanup(self.base_dir().into_owned())
    }

    /// Inserts or overwrites the file at `value` as `name` in the
    /// cache directory.
    ///
    /// Returns the estimated number of files remaining after cleanup
    /// whenever cleanup was initiated.
    ///
    /// Always consumes the file at `value` on success; may consume it
    /// on error.
    fn set(&self, name: &str, value: &Path) -> Result<Option<u64>> {
        let name = validate_file_name(name)?;
        let mut dst = self.base_dir().into_owned();

        let ret = self.maybe_cleanup(&dst)?;
        // Optimistically try to publish the file, without
        // making sure the directory exists: if this works,
        // the directory must have been found.
        dst.push(name);
        if raw_cache::insert_or_update(value, &dst).is_ok() {
            return Ok(ret);
        }

        // We just pushed `name`, we must have a parent.
        std::fs::create_dir_all(dst.parent().expect("must have parent"))?;
        raw_cache::insert_or_update(value, &dst)?;
        Ok(ret)
    }

    /// Inserts the file at `value` as `name` in the cache directory
    /// if there is no such cached entry already, or touches the
    /// cached file if it already exists.
    ///
    /// Returns the estimated number of files remaining after cleanup
    /// whenever cleanup was initiated.
    ///
    /// Always consumes the file at `value` on success; may consume it
    /// on error.
    fn put(&self, name: &str, value: &Path) -> Result<Option<u64>> {
        let name = validate_file_name(name)?;
        let mut dst = self.base_dir().into_owned();

        let ret = self.maybe_cleanup(&dst)?;

        dst.push(name);
        if raw_cache::insert_or_touch(value, &dst).is_ok() {
            return Ok(ret);
        }

        std::fs::create_dir_all(dst.parent().expect("must have parent"))?;
        raw_cache::insert_or_touch(value, &dst)?;
        Ok(ret)
    }

    /// Marks the cached file `name` as newly used, if it exists.
    ///
    /// Returns whether the file `name` exists.
    fn touch(&self, name: &str) -> Result<bool> {
        let name = validate_file_name(name)?;
        let mut target = self.base_dir().into_owned();
        target.push(name);

        raw_cache::touch(&target)
    }
}

#[cfg(test)]
mod test {
    use std::borrow::Cow;
    use std::io::ErrorKind;
    use std::path::Path;
    use std::path::PathBuf;

    use crate::cache_dir::CacheDir;
    use crate::trigger::PeriodicTrigger;

    struct DummyCacheDir {}

    impl CacheDir for DummyCacheDir {
        #[cfg(not(tarpaulin_include))]
        fn temp_dir(&self) -> Cow<Path> {
            unreachable!("should not be called")
        }

        #[cfg(not(tarpaulin_include))]
        fn base_dir(&self) -> Cow<Path> {
            unreachable!("should not be called")
        }

        #[cfg(not(tarpaulin_include))]
        fn trigger(&self) -> &PeriodicTrigger {
            unreachable!("should not be called")
        }

        #[cfg(not(tarpaulin_include))]
        fn capacity(&self) -> usize {
            unreachable!("should not be called")
        }
    }

    // Passing a bad file name to `get` should abort immediately.
    #[test]
    fn test_bad_get() {
        let cache = DummyCacheDir {};

        assert!(matches!(cache.get(""),
                         Err(e) if e.kind() == ErrorKind::InvalidInput));
    }

    // Passing a bad file name to `set` should abort immediately.
    #[test]
    fn test_bad_set() {
        let cache = DummyCacheDir {};

        let path: PathBuf = "/tmp/foo".into();
        assert!(matches!(cache.set(".foo", &path),
                         Err(e) if e.kind() == ErrorKind::InvalidInput));
    }

    // Passing a bad file name to `put` should abort immediately.
    #[test]
    fn test_bad_put() {
        let cache = DummyCacheDir {};

        let path: PathBuf = "/tmp/foo".into();
        assert!(matches!(cache.set("/asd", &path),
                         Err(e) if e.kind() == ErrorKind::InvalidInput));
    }

    // Passing a bad file name to `touch` should abort immediately.
    #[test]
    fn test_bad_touch() {
        let cache = DummyCacheDir {};

        assert!(matches!(cache.touch("\\.test"),
                         Err(e) if e.kind() == ErrorKind::InvalidInput));
    }
}
