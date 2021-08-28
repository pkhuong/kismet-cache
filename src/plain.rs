use std::fs::File;
use std::io::Result;
use std::path::Path;
use std::path::PathBuf;
use std::time::Duration;

use crate::raw_cache;
use crate::trigger::PeriodicTrigger;

/// How many times we want to trigger maintenance per "capacity"
/// inserts.  For example, `MAINTENANCE_SCALE = 3` means we will
/// expect to trigger maintenance after inserting or updating
/// ~capacity / 3 files in the cache.
const MAINTENANCE_SCALE: usize = 3;

/// Put temporary file in this subdirectory of the cache directory.
const TEMP_SUBDIR: &str = ".temp";

/// Delete temporary files with mtime older than this age.
#[cfg(not(test))]
const MAX_TEMP_FILE_AGE: Duration = Duration::from_secs(3600);

// We want a more eager timeout in tests.
#[cfg(test)]
const MAX_TEMP_FILE_AGE: Duration = Duration::from_secs(2);

/// A "plain" cache is a single directory of files.  Given a capacity
/// of `k` files, we will trigger a second chance maintance roughly
/// every `k / 3` (`k / 6` in the long run, given the way
/// `PeriodicTrigger` is implemented) insertions.
#[derive(Clone, Debug)]
pub struct PlainCache {
    // The cached files are siblings of this directory for temporary
    // files.
    temp_dir: PathBuf,

    // Initialised to trigger a second chance maintenance roughly
    // every `capacity / MAINTENANCE_SCALE` cache writes.
    trigger: PeriodicTrigger,

    // The directory has a capacity of roughly this many files;
    // between maintenance, the actual file count may temporarily
    // exceed that capacity.
    capacity: usize,
}

/// Updates the second chance cache state in `base_dir`, and deletes
/// temporary files in that cache directory.
fn cleanup_cache_directory(base_dir: &Path, capacity: usize) -> Result<()> {
    raw_cache::prune(base_dir.to_owned(), capacity)?;

    // Delete old temporary files while we're here.
    let threshold = match std::time::SystemTime::now().checked_sub(MAX_TEMP_FILE_AGE) {
        Some(time) => time,
        None => return Ok(()),
    };

    let mut temp = base_dir.to_owned();
    temp.push(TEMP_SUBDIR);

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

impl PlainCache {
    /// Returns a new cache for approximately `capacity` files in
    /// `base_dir`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `$base_dir/.temp` does not exist and we fail
    /// to create it.
    pub fn new(base_dir: PathBuf, capacity: usize) -> Result<PlainCache> {
        let mut temp_dir = base_dir;

        temp_dir.push(TEMP_SUBDIR);
        std::fs::create_dir_all(&temp_dir)?;

        Ok(PlainCache {
            temp_dir,
            trigger: PeriodicTrigger::new((capacity / MAINTENANCE_SCALE) as u64),
            capacity,
        })
    }

    fn base_dir(&self) -> PathBuf {
        let mut dir = self.temp_dir.clone();

        dir.pop();
        dir
    }

    /// Returns a read-only file for `name` in the cache directory if
    /// it exists, or None if there is no such file.
    ///
    /// Implicitly "touches" the cached file `name` if it exists.
    pub fn get(&self, name: &str) -> Result<Option<File>> {
        let mut target = self.base_dir();
        target.push(name);

        match File::open(&target) {
            Ok(file) => Ok(Some(file)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(e),
        }
    }

    /// Returns a temporary directory suitable for temporary files
    /// that will be published to the cache directory.
    pub fn temp_dir(&self) -> &Path {
        &self.temp_dir
    }

    /// If a periodic cleanup is called for, updates the second chance
    /// cache state and deletes temporary files in that cache directory.
    fn maybe_cleanup(&self, base_dir: &Path) -> Result<()> {
        if self.trigger.event() {
            cleanup_cache_directory(base_dir, self.capacity)
        } else {
            Ok(())
        }
    }

    /// Inserts or overwrites the file at `value` as `name` in the
    /// cache directory.
    ///
    /// Always consumes the file at `value` on success; may consume it
    /// on error.
    pub fn set(&self, name: &str, value: &Path) -> Result<()> {
        let mut dst = self.base_dir();

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
    pub fn put(&self, name: &str, value: &Path) -> Result<()> {
        let mut dst = self.base_dir();

        self.maybe_cleanup(&dst)?;
        dst.push(name);
        raw_cache::insert_or_touch(value, &dst)
    }

    /// Marks the cached file `name` as newly used, if it exists.
    ///
    /// Succeeds if `name` does not exist anymore.
    pub fn touch(&self, name: &str) -> Result<()> {
        let mut target = self.base_dir();
        target.push(name);

        raw_cache::touch(&target)
    }
}

/// Put 20 files in a 10-file cache.  We should find at least 10, but
/// fewer than 20.
#[test]
fn smoke_test() {
    use tempfile::NamedTempFile;
    use test_dir::{DirBuilder, FileType, TestDir};

    // Also leave a file in the temporary subdirectory; we'll check
    // that it gets cleaned up before leaving this function..
    let temp = TestDir::temp()
        .create(TEMP_SUBDIR, FileType::Dir)
        .create(&format!("{}/garbage", TEMP_SUBDIR), FileType::ZeroFile(10));
    // The garbage file must exist.
    assert!(std::fs::metadata(temp.path(&format!("{}/garbage", TEMP_SUBDIR))).is_ok());

    // Make sure the garbage file is old enough to be deleted.
    std::thread::sleep(Duration::from_secs_f64(2.5));
    let cache = PlainCache::new(temp.path("."), 10).expect("::new must succeed");

    for i in 0..20 {
        let name = format!("{}", i);

        let tmp = NamedTempFile::new_in(cache.temp_dir()).expect("new temp file must succeed");
        cache.put(&name, tmp.path()).expect("put must succeed");
    }

    let present: usize = (0..20)
        .map(|i| {
            let name = format!("{}", i);
            cache.get(&name).expect("get must succeed").is_some() as usize
        })
        .sum();

    assert!(present >= 10);
    assert!(present < 20);
    // The temporary garbage file must have been deleted by now.
    assert!(
        matches!(std::fs::metadata(temp.path(&format!("{}/garbage", TEMP_SUBDIR))),
    Err(e) if e.kind() == std::io::ErrorKind::NotFound)
    );
}

/// Publish a file, make sure we can read it, then overwrite, and
/// confirm that the new contents are visible.
#[test]
fn test_set() {
    use std::io::{Read, Write};
    use tempfile::NamedTempFile;
    use test_dir::{DirBuilder, TestDir};

    let temp = TestDir::temp();
    let cache = PlainCache::new(temp.path("."), 1).expect("::new must succeed");

    {
        let tmp = NamedTempFile::new_in(cache.temp_dir()).expect("new temp file must succeed");
        tmp.as_file().write_all(b"v1").expect("write must succeed");

        cache
            .set("entry", tmp.path())
            .expect("initial set must succeed");
    }

    {
        let mut cached = cache
            .get("entry")
            .expect("must succeed")
            .expect("must be found");
        let mut dst = Vec::new();
        cached.read_to_end(&mut dst).expect("read must succeed");
        assert_eq!(&dst, b"v1");
    }

    // Now overwrite; it should take.
    {
        let tmp = NamedTempFile::new_in(cache.temp_dir()).expect("new temp file must succeed");
        tmp.as_file().write_all(b"v2").expect("write must succeed");

        cache
            .set("entry", tmp.path())
            .expect("overwrite must succeed");
    }

    {
        let mut cached = cache
            .get("entry")
            .expect("must succeed")
            .expect("must be found");
        let mut dst = Vec::new();
        cached.read_to_end(&mut dst).expect("read must succeed");
        assert_eq!(&dst, b"v2");
    }
}

/// Publish a file, make sure we can read it, and make sure that a
/// second put does not update its contents.
#[test]
fn test_put() {
    use std::io::{Read, Write};
    use tempfile::NamedTempFile;
    use test_dir::{DirBuilder, TestDir};

    let temp = TestDir::temp();
    let cache = PlainCache::new(temp.path("."), 1).expect("::new must succeed");

    {
        let tmp = NamedTempFile::new_in(cache.temp_dir()).expect("new temp file must succeed");
        tmp.as_file().write_all(b"v1").expect("write must succeed");

        cache
            .put("entry", tmp.path())
            .expect("initial set must succeed");
    }

    {
        let mut cached = cache
            .get("entry")
            .expect("must succeed")
            .expect("must be found");
        let mut dst = Vec::new();
        cached.read_to_end(&mut dst).expect("read must succeed");
        assert_eq!(&dst, b"v1");
    }

    // Now put again; it shouldn't overwrite.
    {
        let tmp = NamedTempFile::new_in(cache.temp_dir()).expect("new temp file must succeed");
        tmp.as_file().write_all(b"v2").expect("write must succeed");

        cache
            .put("entry", tmp.path())
            .expect("overwrite must succeed");
    }

    {
        let mut cached = cache
            .get("entry")
            .expect("must succeed")
            .expect("must be found");
        let mut dst = Vec::new();
        cached.read_to_end(&mut dst).expect("read must succeed");
        assert_eq!(&dst, b"v1");
    }
}

/// Keep publishing new files, but also always touch the first.
/// That first file should never be deleted.
#[test]
fn test_touch() {
    use tempfile::NamedTempFile;
    use test_dir::{DirBuilder, TestDir};

    let temp = TestDir::temp();
    let cache = PlainCache::new(temp.path("."), 5).expect("::new must succeed");

    for i in 0..15 {
        let name = format!("{}", i);

        cache.touch("0").expect("touch must not fail");

        let tmp = NamedTempFile::new_in(cache.temp_dir()).expect("new temp file must succeed");
        cache.put(&name, tmp.path()).expect("put must succeed");
        // Make sure enough time elapses for the next file to get
        // a different timestamp.
        std::thread::sleep(Duration::from_secs_f64(1.5));
    }

    // We should still find "0": it's the oldest, but we also keep
    // touching it.
    cache.get("0").expect("must succed").expect("must be found");
}

/// Trigger a cleanup while a very recent file is still in the
/// temporary subdirectory.  It should remain there.
#[test]
fn test_recent_temp_file() {
    use tempfile::NamedTempFile;
    use test_dir::{DirBuilder, FileType, TestDir};

    // Also leave a file in the temporary subdirectory; we'll check
    // that it gets cleaned up before leaving this function..
    let temp = TestDir::temp()
        .create(TEMP_SUBDIR, FileType::Dir)
        .create(&format!("{}/garbage", TEMP_SUBDIR), FileType::ZeroFile(10));
    // The garbage file must exist.
    assert!(std::fs::metadata(temp.path(&format!("{}/garbage", TEMP_SUBDIR))).is_ok());

    let cache = PlainCache::new(temp.path("."), 1).expect("::new must succeed");

    for i in 0..2 {
        let tmp = NamedTempFile::new_in(cache.temp_dir()).expect("new temp file must succeed");
        cache
            .put(&format!("{}", i), tmp.path())
            .expect("put must succeed");
    }

    // The garbage file must still exist.
    assert!(std::fs::metadata(temp.path(&format!("{}/garbage", TEMP_SUBDIR))).is_ok());
}
