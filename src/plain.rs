use std::borrow::Cow;
use std::fs::File;
use std::io::Result;
use std::path::Path;
use std::path::PathBuf;

use crate::cache_dir::CacheDir;
use crate::trigger::PeriodicTrigger;

/// How many times we want to trigger maintenance per "capacity"
/// inserts.  For example, `MAINTENANCE_SCALE = 3` means we will
/// expect to trigger maintenance after inserting or updating
/// ~capacity / 3 files in the cache.
const MAINTENANCE_SCALE: usize = 3;

/// Put temporary file in this subdirectory of the cache directory.
const TEMP_SUBDIR: &str = ".temp";

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

impl CacheDir for PlainCache {
    #[inline]
    fn temp_dir(&self) -> Cow<Path> {
        Cow::from(&self.temp_dir)
    }

    #[inline]
    fn base_dir(&self) -> Cow<Path> {
        Cow::from(self.temp_dir.parent().unwrap_or(&self.temp_dir))
    }

    #[inline]
    fn trigger(&self) -> &PeriodicTrigger {
        &self.trigger
    }

    #[inline]
    fn capacity(&self) -> usize {
        self.capacity
    }
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

    /// Returns a read-only file for `name` in the cache directory if
    /// it exists, or None if there is no such file.
    ///
    /// Implicitly "touches" the cached file `name` if it exists.
    pub fn get(&self, name: &str) -> Result<Option<File>> {
        CacheDir::get(self, name)
    }

    /// Returns a temporary directory suitable for temporary files
    /// that will be published to the cache directory.
    pub fn temp_dir(&self) -> Cow<Path> {
        CacheDir::temp_dir(self)
    }

    /// Inserts or overwrites the file at `value` as `name` in the
    /// cache directory.
    ///
    /// Always consumes the file at `value` on success; may consume it
    /// on error.
    pub fn set(&self, name: &str, value: &Path) -> Result<()> {
        CacheDir::set(self, name, value)?;
        Ok(())
    }

    /// Inserts the file at `value` as `name` in the cache directory
    /// if there is no such cached entry already, or touches the
    /// cached file if it already exists.
    ///
    /// Always consumes the file at `value` on success; may consume it
    /// on error.
    pub fn put(&self, name: &str, value: &Path) -> Result<()> {
        CacheDir::put(self, name, value)?;
        Ok(())
    }

    /// Marks the cached file `name` as newly used, if it exists.
    ///
    /// Returns whether `name` exists.
    pub fn touch(&self, name: &str) -> Result<bool> {
        CacheDir::touch(self, name)
    }
}

/// Put 20 files in a 10-file cache.  We should find at least 10, but
/// fewer than 20, and their contents should match.
#[test]
fn smoke_test() {
    use tempfile::NamedTempFile;
    use test_dir::{DirBuilder, FileType, TestDir};

    // The payload for file `i` is `PAYLOAD_MULTIPLIER * i`.
    const PAYLOAD_MULTIPLIER: usize = 13;

    // Also leave a file in the temporary subdirectory; we'll check
    // that it gets cleaned up before leaving this function..
    let temp = TestDir::temp()
        .create(TEMP_SUBDIR, FileType::Dir)
        .create(&format!("{}/garbage", TEMP_SUBDIR), FileType::ZeroFile(10));
    // The garbage file must exist.
    assert!(std::fs::metadata(temp.path(&format!("{}/garbage", TEMP_SUBDIR))).is_ok());

    // Make sure the garbage file is old enough to be deleted.
    std::thread::sleep(std::time::Duration::from_secs_f64(2.5));
    let cache = PlainCache::new(temp.path("."), 10).expect("::new must succeed");

    for i in 0..20 {
        let name = format!("{}", i);

        let tmp = NamedTempFile::new_in(cache.temp_dir()).expect("new temp file must succeed");
        std::fs::write(tmp.path(), format!("{}", PAYLOAD_MULTIPLIER * i))
            .expect("write must succeed");
        cache.put(&name, tmp.path()).expect("put must succeed");
    }

    let present: usize = (0..20)
        .map(|i| {
            let name = format!("{}", i);
            match cache.get(&name).expect("get must succeed") {
                Some(mut file) => {
                    use std::io::Read;
                    let mut buf = Vec::new();
                    file.read_to_end(&mut buf).expect("read must succeed");
                    assert_eq!(buf, format!("{}", PAYLOAD_MULTIPLIER * i).into_bytes());
                    1
                }
                None => 0,
            }
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

        // After the first write, touch should find our file.
        assert_eq!(cache.touch("0").expect("touch must not fail"), i > 0);

        let tmp = NamedTempFile::new_in(cache.temp_dir()).expect("new temp file must succeed");
        cache.put(&name, tmp.path()).expect("put must succeed");
        // Make sure enough time elapses for the first file to get
        // an older timestamp than the rest.
        if i == 0 {
            std::thread::sleep(std::time::Duration::from_secs_f64(1.5));
        }
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