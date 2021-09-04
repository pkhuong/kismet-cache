//! A `ShardedCache` uses the same basic file-based second chance
//! strategy as a `PlainCache`.  However, while the simple plain cache
//! is well suited to small caches (down to 2-3 files, and up maybe
//! one hundred), this sharded version can scale nearly arbitrarily
//! high: each shard should have fewer than one hundred or so files,
//! but there may be arbitrarily many shards (up to filesystem limits,
//! since each shard is a subdirectory).
use std::borrow::Cow;
use std::fs::File;
use std::io::Result;
use std::path::Path;
use std::path::PathBuf;

use crate::cache_dir::CacheDir;
use crate::trigger::PeriodicTrigger;

/// We will aim to trigger maintenance at least `MAINTENANCE_SCALE`
/// times per total capacity inserts or updates, and at least once per
/// shard capacity inserts or updates.
const MAINTENANCE_SCALE: usize = 4;

/// Put temporary file in this subdirectory of the cache directory.
const TEMP_SUBDIR: &str = ".temp";

const RANDOM_MULTIPLIER: u64 = 0xf2efdf1111adba6f;

/// Cache keys consist of a filename and a hash value.  The hash
/// function must be identical for all processes that access the same
/// sharded cache directory.
#[derive(Clone, Copy, Debug)]
pub struct Key<'a> {
    pub name: &'a str,
    pub hash: u64,
}

impl<'a> Key<'a> {
    pub fn new(name: &str, hash: u64) -> Key {
        Key { name, hash }
    }
}

/// A sharded cache is a hash-sharded directory of cache
/// subdirectories.  Each subdirectory is managed as an
/// independent second chance cache directory.
#[derive(Clone, Debug)]
pub struct ShardedCache {
    // The parent directory for each shard (cache subdirectory).
    base_dir: PathBuf,
    // Triggers periodic second chance maintenance.  It is set to the
    // least (most frequent) period between ~1/4 the total capacity,
    // and each shard's capacity.
    trigger: PeriodicTrigger,
    // Number of shards in the cache.
    num_shards: usize,
    // Capacity for each shard (rounded up to an integer).
    shard_capacity: usize,
}

#[inline]
fn format_id(shard: usize) -> String {
    format!("{:04x}", shard)
}

/// We create short-lived Shard objects whenever we want to work with
/// a given shard of the sharded cache dir.
struct Shard {
    shard_dir: PathBuf,
    trigger: PeriodicTrigger,
    capacity: usize,
}

impl CacheDir for Shard {
    #[inline]
    fn temp_dir(&self) -> Cow<Path> {
        let mut dir = self.shard_dir.clone();
        dir.push(TEMP_SUBDIR);
        Cow::from(dir)
    }

    #[inline]
    fn base_dir(&self) -> Cow<Path> {
        Cow::from(&self.shard_dir)
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

impl ShardedCache {
    /// Returns a new cache for approximately `total_capacity` files,
    /// stores in `num_shards` subdirectories of `base_dir`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if we fail to create any of the cache
    /// subdirectories and their `.temp` sub-subdirectories.
    pub fn new(
        mut base_dir: PathBuf,
        mut num_shards: usize,
        mut total_capacity: usize,
    ) -> Result<ShardedCache> {
        if num_shards == 0 {
            num_shards = 1;
        }

        if total_capacity < num_shards {
            total_capacity = num_shards;
        }

        // Create the directory structure eagerly.
        for i in 0..num_shards {
            base_dir.push(&format_id(i));
            base_dir.push(TEMP_SUBDIR);

            std::fs::create_dir_all(&base_dir)?;

            base_dir.pop();
            base_dir.pop();
        }

        let shard_capacity =
            (total_capacity / num_shards) + ((total_capacity % num_shards) != 0) as usize;
        let trigger =
            PeriodicTrigger::new(shard_capacity.min(total_capacity / MAINTENANCE_SCALE) as u64);

        Ok(ShardedCache {
            base_dir,
            trigger,
            num_shards,
            shard_capacity,
        })
    }

    /// Returns a random shard id.
    fn random_shard_id(&self) -> usize {
        use rand::Rng;

        rand::thread_rng().gen_range(0..self.num_shards)
    }

    /// Returns the shard id for `key`.
    fn shard_id(&self, key: Key) -> usize {
        // We can't assume the hash is well distributed, so mix it
        // around a bit with a multiplicative hash.
        let hash = key.hash.wrapping_mul(RANDOM_MULTIPLIER) as u128;

        // Map the hashed hash to a shard id with a fixed point
        // multiplication.
        ((self.num_shards as u128 * hash) >> 64) as usize
    }

    /// Returns a shard object for the `shard_id`.
    fn shard(&self, shard_id: usize) -> Shard {
        let mut dir = self.base_dir.clone();
        dir.push(&format_id(shard_id));
        Shard {
            shard_dir: dir,
            trigger: self.trigger,
            capacity: self.shard_capacity,
        }
    }

    /// Returns a read-only file for `key` in the shard cache
    /// directory if it exists, or None if there is no such file.
    ///
    /// Implicitly "touches" the cached file if it exists.
    pub fn get(&self, key: Key) -> Result<Option<File>> {
        self.shard(self.shard_id(key)).get(key.name)
    }

    /// Returns a temporary directory suitable for temporary files
    /// that will be published to the shard cache directory.
    ///
    /// When this temporary file will be published at a known `Key`,
    /// populate `key` for improved behaviour.
    pub fn temp_dir(&self, key: Option<Key>) -> Result<PathBuf> {
        let shard_id = match key {
            Some(key) => self.shard_id(key),
            None => self.random_shard_id(),
        };
        let shard = self.shard(shard_id);
        if self.trigger.event() {
            shard.cleanup_temp_directory()?;
        }

        Ok(shard.temp_dir().into_owned())
    }

    /// Inserts or overwrites the file at `value` as `key` in the
    /// sharded cache directory.
    ///
    /// Always consumes the file at `value` on success; may consume it
    /// on error.
    pub fn set(&self, key: Key, value: &Path) -> Result<()> {
        self.shard(self.shard_id(key)).set(key.name, value)?;
        Ok(())
    }

    /// Inserts the file at `value` as `key` in the cache directory
    /// if there is no such cached entry already, or touches the
    /// cached file if it already exists.
    ///
    /// Always consumes the file at `value` on success; may consume it
    /// on error.
    pub fn put(&self, key: Key, value: &Path) -> Result<()> {
        self.shard(self.shard_id(key)).put(key.name, value)?;
        Ok(())
    }

    /// Marks the cached file `key` as newly used, if it exists.
    ///
    /// Succeeds even if `key` does not exist.
    pub fn touch(&self, key: Key) -> Result<()> {
        self.shard(self.shard_id(key)).touch(key.name)
    }
}

/// Put 40 files in a 3x3-file cache.  We should find at least 9, but
/// fewer than 40, and their contents should match.
#[test]
fn smoke_test() {
    use tempfile::NamedTempFile;
    use test_dir::{DirBuilder, TestDir};

    // The payload for file `i` is `PAYLOAD_MULTIPLIER * i`.
    const PAYLOAD_MULTIPLIER: usize = 113;

    let temp = TestDir::temp();
    let cache = ShardedCache::new(temp.path("."), 3, 9).expect("::new must succeed");

    for i in 0..40 {
        let name = format!("{}", i);

        let temp_dir = cache.temp_dir(None).expect("temp_dir must succeed");
        let tmp = NamedTempFile::new_in(temp_dir).expect("new temp file must succeed");
        std::fs::write(tmp.path(), format!("{}", PAYLOAD_MULTIPLIER * i))
            .expect("write must succeed");
        cache
            .put(Key::new(&name, i as u64), tmp.path())
            .expect("put must succeed");
    }

    let present: usize = (0..40)
        .map(|i| {
            let name = format!("{}", i);
            match cache
                .get(Key::new(&name, i as u64))
                .expect("get must succeed")
            {
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

    assert!(present >= 9);
    assert!(present < 40);
}

/// Publish a file, make sure we can read it, then overwrite, and
/// confirm that the new contents are visible.
#[test]
fn test_set() {
    use std::io::{Read, Write};
    use tempfile::NamedTempFile;
    use test_dir::{DirBuilder, TestDir};

    let temp = TestDir::temp();
    let cache = ShardedCache::new(temp.path("."), 0, 0).expect("::new must succeed");

    {
        let tmp = NamedTempFile::new_in(cache.temp_dir(None).expect("temp_dir must succeed"))
            .expect("new temp file must succeed");
        tmp.as_file().write_all(b"v1").expect("write must succeed");

        cache
            .set(Key::new("entry", 1), tmp.path())
            .expect("initial set must succeed");
    }

    {
        let mut cached = cache
            .get(Key::new("entry", 1))
            .expect("must succeed")
            .expect("must be found");
        let mut dst = Vec::new();
        cached.read_to_end(&mut dst).expect("read must succeed");
        assert_eq!(&dst, b"v1");
    }

    // Now overwrite; it should take.
    {
        let tmp = NamedTempFile::new_in(cache.temp_dir(None).expect("temp_dir must succeed"))
            .expect("new temp file must succeed");
        tmp.as_file().write_all(b"v2").expect("write must succeed");

        cache
            .set(Key::new("entry", 1), tmp.path())
            .expect("overwrite must succeed");
    }

    {
        let mut cached = cache
            .get(Key::new("entry", 1))
            .expect("must succeed")
            .expect("must be found");
        let mut dst = Vec::new();
        cached.read_to_end(&mut dst).expect("read must succeed");
        assert_eq!(&dst, b"v2");
    }
}

/// Publish a file, fail to put a new one with different data, and
/// confirm that the old contents are visible.
#[test]
fn test_put() {
    use std::io::{Read, Write};
    use tempfile::NamedTempFile;
    use test_dir::{DirBuilder, TestDir};

    let temp = TestDir::temp();
    let cache = ShardedCache::new(temp.path("."), 0, 0).expect("::new must succeed");

    {
        let tmp = NamedTempFile::new_in(cache.temp_dir(None).expect("temp_dir must succeed"))
            .expect("new temp file must succeed");
        tmp.as_file().write_all(b"v1").expect("write must succeed");

        cache
            .set(Key::new("entry", 1), tmp.path())
            .expect("initial set must succeed");
    }

    // Now put; it should not take.
    {
        let tmp = NamedTempFile::new_in(cache.temp_dir(None).expect("temp_dir must succeed"))
            .expect("new temp file must succeed");
        tmp.as_file().write_all(b"v2").expect("write must succeed");

        cache
            .put(Key::new("entry", 1), tmp.path())
            .expect("put must succeed");
    }

    {
        let mut cached = cache
            .get(Key::new("entry", 1))
            .expect("must succeed")
            .expect("must be found");
        let mut dst = Vec::new();
        cached.read_to_end(&mut dst).expect("read must succeed");
        assert_eq!(&dst, b"v1");
    }
}

/// Put 2000 files in a 2x300-file cache, and keep touching the first.
/// We should always find the first file, even after all that cleanup.
#[test]
fn test_touch() {
    use std::io::Read;
    use tempfile::NamedTempFile;
    use test_dir::{DirBuilder, TestDir};

    // The payload for file `i` is `PAYLOAD_MULTIPLIER * i`.
    const PAYLOAD_MULTIPLIER: usize = 113;

    let temp = TestDir::temp();
    let cache = ShardedCache::new(temp.path("."), 2, 600).expect("::new must succeed");

    for i in 0..2000 {
        cache.touch(Key::new("0", 0)).expect("touch must succeed");

        let name = format!("{}", i);

        let temp_dir = cache.temp_dir(None).expect("temp_dir must succeed");
        let tmp = NamedTempFile::new_in(temp_dir).expect("new temp file must succeed");
        std::fs::write(tmp.path(), format!("{}", PAYLOAD_MULTIPLIER * i))
            .expect("write must succeed");
        cache
            .put(Key::new(&name, i as u64), tmp.path())
            .expect("put must succeed");
        if i == 0 {
            // Make sure file "0" is measurably older than the others.
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
    }

    let mut file = cache
        .get(Key::new("0", 0))
        .expect("get must succeed")
        .expect("file must be found");
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).expect("read must succeed");
    assert_eq!(buf, b"0");
}
