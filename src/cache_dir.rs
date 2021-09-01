use std::borrow::Cow;
use std::fs::File;
use std::io::Result;
use std::path::Path;
use std::time::Duration;

use crate::raw_cache;
use crate::trigger::PeriodicTrigger;

/// Delete temporary files with mtime older than this age.
#[cfg(not(test))]
const MAX_TEMP_FILE_AGE: Duration = Duration::from_secs(3600);

// We want a more eager timeout in tests.
#[cfg(test)]
const MAX_TEMP_FILE_AGE: Duration = Duration::from_secs(2);

/// Deletes any file with mtime older than `MAX_TEMP_FILE_AGE` in
/// `temp_dir`.
fn cleanup_temporary_directory(temp_dir: Cow<Path>) -> Result<()> {
    let threshold = match std::time::SystemTime::now().checked_sub(MAX_TEMP_FILE_AGE) {
        Some(time) => time,
        None => return Ok(()),
    };

    let mut temp = temp_dir.into_owned();
    for dirent in std::fs::read_dir(&temp)?.flatten() {
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

    /// Returns a read-only file for `name` in the cache directory if
    /// it exists, or None if there is no such file.
    ///
    /// Implicitly "touches" the cached file `name` if it exists.
    fn get(&self, name: &str) -> Result<Option<File>> {
        let mut target = self.base_dir().into_owned();
        target.push(name);

        match File::open(&target) {
            Ok(file) => Ok(Some(file)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Deletes old files in this cache directory's subdirectory of
    /// temporary files.
    fn cleanup_temp_directory(&self) -> Result<()> {
        cleanup_temporary_directory(self.temp_dir())
    }

    /// If a periodic cleanup is called for, updates the second chance
    /// cache state and deletes temporary files in that cache directory.
    ///
    /// Returns true whenever cleanup was initiated.
    fn maybe_cleanup(&self, base_dir: &Path) -> Result<bool> {
        if self.trigger().event() {
            raw_cache::prune(base_dir.to_owned(), self.capacity())?;
            // Delete old temporary files while we're here.
            self.cleanup_temp_directory()?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Inserts or overwrites the file at `value` as `name` in the
    /// cache directory.
    ///
    /// Always consumes the file at `value` on success; may consume it
    /// on error.
    fn set(&self, name: &str, value: &Path) -> Result<()> {
        let mut dst = self.base_dir().into_owned();

        self.maybe_cleanup(&dst)?;
        dst.push(name);
        raw_cache::insert_or_update(value, &dst)
    }

    /// Inserts the file at `value` as `name` in the cache directory
    /// if there is no such cached entry already, or touches the
    /// cached file if it already exists.
    ///
    /// Always consumes the file at `value` on success; may consume it
    /// on error.
    fn put(&self, name: &str, value: &Path) -> Result<()> {
        let mut dst = self.base_dir().into_owned();

        self.maybe_cleanup(&dst)?;
        dst.push(name);
        raw_cache::insert_or_touch(value, &dst)
    }

    /// Marks the cached file `name` as newly used, if it exists.
    ///
    /// Succeeds if `name` does not exist anymore.
    fn touch(&self, name: &str) -> Result<()> {
        let mut target = self.base_dir().into_owned();
        target.push(name);

        raw_cache::touch(&target)
    }
}
