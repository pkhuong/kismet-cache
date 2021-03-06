//! A [`crate::sharded::Cache`] uses the same basic file-based second
//! chance strategy as a [`crate::plain::Cache`].  However, while the
//! simple plain cache is well suited to small caches (down to 2-3
//! files, and up maybe one hundred), this sharded version can scale
//! nearly arbitrarily high: each shard should have fewer than one
//! hundred or so files, but there may be arbitrarily many shards (up
//! to filesystem limits, since each shard is a subdirectory).
//!
//! A sharded cache directory consists of shard subdirectories (with
//! name equal to the shard index printed as `%04x`), each of which
//! contains the cached files for that shard, under their `key` name,
//! and an optional `.kismet_temp` subdirectory for temporary files.
//!
//! This module is useful for lower level usage; in most cases, the
//! [`crate::Cache`] is more convenient and just as efficient.  In
//! particular, a `crate::sharded::Cache` *does not* invoke
//! [`std::fs::File::sync_all`] or [`std::fs::File::sync_data`]: the
//! caller should sync files before letting Kismet persist them in a
//! directory, if necessary.
//!
//! The cache's contents will grow past its stated capacity, but
//! should rarely reach more than twice that capacity, especially
//! when the shard capacity is less than 128 files.
use std::borrow::Cow;
use std::fs::File;
use std::io::Result;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::Arc;

use crate::cache_dir::CacheDir;
use crate::multiplicative_hash::MultiplicativeHash;
use crate::trigger::PeriodicTrigger;
use crate::Key;
use crate::KISMET_TEMPORARY_SUBDIRECTORY as TEMP_SUBDIR;

/// We will aim to trigger maintenance at least `MAINTENANCE_SCALE`
/// times per total capacity inserts or updates, and at least once per
/// shard capacity inserts or updates.
const MAINTENANCE_SCALE: usize = 2;

/// These mixers must be the same for all processes that access the
/// same sharded cache directory.  That's why we derive the parameters
/// in a const function.
const PRIMARY_MIXER: MultiplicativeHash =
    MultiplicativeHash::new_keyed(b"kismet: primary shard mixer");

const SECONDARY_MIXER: MultiplicativeHash =
    MultiplicativeHash::new_keyed(b"kismet: secondary shard mixer");

/// A sharded cache is a hash-sharded directory of cache
/// subdirectories.  Each subdirectory is managed as an
/// independent second chance cache directory.
#[derive(Clone, Debug)]
pub struct Cache {
    // The current load (number of files) estimate for each shard.
    load_estimates: Arc<[AtomicU8]>,
    // The parent directory for each shard (cache subdirectory).
    base_dir: PathBuf,
    // Triggers periodic second chance maintenance.  It is set to the
    // least (most frequent) period between ~1/2 the total capacity,
    // and each shard's capacity.  Whenever the `trigger` fires, we
    // will maintain two different shards: the one we just updated,
    // and another randomly chosen shard.
    trigger: PeriodicTrigger,
    // Number of shards in the cache, at least 2.
    num_shards: usize,
    // Capacity for each shard (rounded up to an integer), at least 1.
    shard_capacity: usize,
}

/// Converts a shard id to a subdirectory name.
///
/// We use a dot prefix because the resulting subdirectory names are
/// guaranteed not to collide with "plain" cache filenames.  This
/// means we can switch between the sharded and plain (unsharded)
/// strategy for the same directory, without any chance of
/// misinterpreted file name.
#[inline]
fn format_id(shard: usize) -> String {
    format!(".kismet_{:04x}", shard)
}

/// We create short-lived Shard objects whenever we want to work with
/// a given shard of the sharded cache dir.
struct Shard {
    id: usize,
    shard_dir: PathBuf,
    trigger: PeriodicTrigger,
    capacity: usize,
}

impl Shard {
    /// Returns a shard object for a new shard `id`.
    fn replace_shard(self, id: usize) -> Shard {
        let mut shard_dir = self.shard_dir;
        shard_dir.pop();
        shard_dir.push(&format_id(id));
        Shard {
            id,
            shard_dir,
            trigger: self.trigger,
            capacity: self.capacity,
        }
    }

    /// Returns whether the file `name` exists in this shard.
    fn file_exists(&mut self, name: &str) -> bool {
        self.shard_dir.push(name);
        let result = std::fs::metadata(&self.shard_dir);
        self.shard_dir.pop();

        result.is_ok()
    }
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

impl Cache {
    /// Returns a new cache for approximately `total_capacity` files,
    /// stores in `num_shards` subdirectories of `base_dir`.
    pub fn new(base_dir: PathBuf, mut num_shards: usize, mut total_capacity: usize) -> Cache {
        // We assume at least two shards.
        if num_shards < 2 {
            num_shards = 2;
        }

        if total_capacity < num_shards {
            total_capacity = num_shards;
        }

        let mut load_estimates = Vec::with_capacity(num_shards);
        load_estimates.resize_with(num_shards, || AtomicU8::new(0));
        let shard_capacity =
            (total_capacity / num_shards) + ((total_capacity % num_shards) != 0) as usize;
        let trigger =
            PeriodicTrigger::new(shard_capacity.min(total_capacity / MAINTENANCE_SCALE) as u64);

        Cache {
            load_estimates: load_estimates.into_boxed_slice().into(),
            base_dir,
            trigger,
            num_shards,
            shard_capacity,
        }
    }

    /// Returns a random shard id.
    fn random_shard_id(&self) -> usize {
        use rand::Rng;

        rand::thread_rng().gen_range(0..self.num_shards)
    }

    /// Given shard ids `base` and `other`, returns a new shard id for
    /// `other` such that `base` and `other` do not collide.
    fn other_shard_id(&self, base: usize, mut other: usize) -> usize {
        if base != other {
            return other;
        }

        other += 1;
        if other < self.num_shards {
            other
        } else {
            0
        }
    }

    /// Returns the two shard ids for `key`.
    fn shard_ids(&self, key: Key) -> (usize, usize) {
        // We can't assume the hash is well distributed, so mix it
        // around a bit with a multiplicative hash.
        let h1 = PRIMARY_MIXER.map(key.hash, self.num_shards);
        let h2 = SECONDARY_MIXER.map(key.secondary_hash, self.num_shards);

        // We do not apply a 2-left strategy because our load
        // estimates can saturate.  When that happens, we want to
        // revert to sharding based on `key.hash`.
        (h1, self.other_shard_id(h1, h2))
    }

    /// Reorders two shard ids to return the least loaded first.
    fn sort_by_load(&self, (h1, h2): (usize, usize)) -> (usize, usize) {
        let load1 = self.load_estimates[h1].load(Relaxed) as usize;
        let load2 = self.load_estimates[h2].load(Relaxed) as usize;

        // Clamp loads at the shard capacity: when both shards are
        // over the capacity, they're equally overloaded.  This also
        // lets us revert to only using `key.hash` when at capacity.
        let capacity = self.shard_capacity;
        if load1.clamp(0, capacity) <= load2.clamp(0, capacity) {
            (h1, h2)
        } else {
            (h2, h1)
        }
    }

    /// Returns a shard object for the `shard_id`.
    fn shard(&self, shard_id: usize) -> Shard {
        let mut dir = self.base_dir.clone();
        dir.push(&format_id(shard_id));
        Shard {
            id: shard_id,
            shard_dir: dir,
            trigger: self.trigger,
            capacity: self.shard_capacity,
        }
    }

    /// Returns a read-only file for `key` in the shard cache
    /// directory if it exists, or None if there is no such file.
    /// Fails with `ErrorKind::InvalidInput` if `key.name` is invalid
    /// (empty, or starts with a dot or a forward or back slash).
    ///
    /// Implicitly "touches" the cached file if it exists.
    pub fn get(&self, key: Key) -> Result<Option<File>> {
        let (h1, h2) = self.shard_ids(key);
        let shard = self.shard(h1);

        if let Some(file) = shard.get(key.name)? {
            Ok(Some(file))
        } else {
            shard.replace_shard(h2).get(key.name)
        }
    }

    /// Returns a temporary directory suitable for temporary files
    /// that will be published to the shard cache directory.
    ///
    /// When this temporary file will be published at a known `Key`,
    /// populate `key` for improved behaviour.
    pub fn temp_dir(&self, key: Option<Key>) -> Result<Cow<Path>> {
        let shard_id = match key {
            Some(key) => self.sort_by_load(self.shard_ids(key)).0,
            None => self.random_shard_id(),
        };
        let shard = self.shard(shard_id);
        if self.trigger.event() {
            shard.cleanup_temp_directory()?;
        }

        Ok(Cow::from(shard.ensure_temp_dir()?.into_owned()))
    }

    /// Updates the load estimate for `shard_id` with the value
    /// returned by `CacheDir::{set,put}`.
    fn update_estimate(&self, shard_id: usize, update: Option<u64>) {
        let target = &self.load_estimates[shard_id];
        match update {
            // If we have an updated estimate, overwrite what we have,
            // and take the newly added file into account.
            Some(remaining) => {
                let update = remaining.clamp(0, u8::MAX as u64 - 1) as u8;
                target.store(update + 1, Relaxed);
            }
            // Otherwise, increment by one with saturation.
            None => {
                let _ = target.fetch_update(Relaxed, Relaxed, |i| {
                    if i < u8::MAX {
                        Some(i + 1)
                    } else {
                        None
                    }
                });
            }
        };
    }

    /// Performs a second chance maintenance on `shard`.
    fn force_maintain_shard(&self, shard: Shard) -> Result<()> {
        let update = shard.maintain()?.clamp(0, u8::MAX as u64) as u8;
        self.load_estimates[shard.id].store(update, Relaxed);
        Ok(())
    }

    /// Performs a second chance maintenance on a randomly chosen shard
    /// that is not `base`.
    fn maintain_random_other_shard(&self, base: Shard) -> Result<()> {
        let shard_id = self.other_shard_id(base.id, self.random_shard_id());
        self.force_maintain_shard(base.replace_shard(shard_id))
    }

    /// Inserts or overwrites the file at `value` as `key` in the
    /// sharded cache directory.  There may be two entries for the
    /// same key with concurrent `set` or `put` calls.  Fails with
    /// `ErrorKind::InvalidInput` if `key.name` is invalid (empty, or
    /// starts with a dot or a forward or back slash).
    ///
    /// Always consumes the file at `value` on success; may consume it
    /// on error.
    pub fn set(&self, key: Key, value: &Path) -> Result<()> {
        let (h1, h2) = self.sort_by_load(self.shard_ids(key));
        let mut shard = self.shard(h2);

        // If the file does not already exist in the secondary shard,
        // use the primary.
        if !shard.file_exists(key.name) {
            shard = shard.replace_shard(h1);
        }

        let update = shard.set(key.name, value)?;
        self.update_estimate(h1, update);

        // If we performed maintenance on this shard, also maintain
        // a second random shard: writes might be concentrated on a
        // few shard, but we can still spread the love, if only to
        // clean up temporary files.
        if update.is_some() {
            self.maintain_random_other_shard(shard)?;
        } else if self.load_estimates[h1].load(Relaxed) as usize / 2 > self.shard_capacity {
            // Otherwise, we can also force a maintenance for this
            // shard if we're pretty sure it has grown much too big.
            self.force_maintain_shard(shard)?;
        }

        Ok(())
    }

    /// Inserts the file at `value` as `key` in the cache directory if
    /// there is no such cached entry already, or touches the cached
    /// file if it already exists.  There may be two entries for the
    /// same key with concurrent `set` or `put` calls.  Fails with
    /// `ErrorKind::InvalidInput` if `key.name` is invalid (empty, or
    /// starts with a dot or a forward or back slash).
    ///
    /// Always consumes the file at `value` on success; may consume it
    /// on error.
    pub fn put(&self, key: Key, value: &Path) -> Result<()> {
        let (h1, h2) = self.sort_by_load(self.shard_ids(key));
        let mut shard = self.shard(h2);

        // If the file does not already exist in the secondary shard,
        // use the primary.
        if !shard.file_exists(key.name) {
            shard = shard.replace_shard(h1);
        }

        let update = shard.put(key.name, value)?;
        self.update_estimate(h1, update);

        // If we performed maintenance on this shard, also maintain
        // a second random shard.
        if update.is_some() {
            self.maintain_random_other_shard(shard)?;
        } else if self.load_estimates[h1].load(Relaxed) as usize / 2 > self.shard_capacity {
            self.force_maintain_shard(shard)?;
        }

        Ok(())
    }

    /// Marks the cached file `key` as newly used, if it exists.
    /// Fails with `ErrorKind::InvalidInput` if `key.name` is invalid
    /// (empty, or starts with a dot or a forward or back slash).
    ///
    /// Returns whether a file for `key` exists in the cache.
    pub fn touch(&self, key: Key) -> Result<bool> {
        let (h1, h2) = self.shard_ids(key);
        let shard = self.shard(h1);

        if shard.touch(key.name)? {
            return Ok(true);
        }

        shard.replace_shard(h2).touch(key.name)
    }
}

/// Put 200 files in a 3x3-file cache.  We should find at least 9, but
/// at most 18 (2x the capacity), and their contents should match.
#[test]
fn smoke_test() {
    use tempfile::NamedTempFile;
    use test_dir::{DirBuilder, TestDir};

    // The payload for file `i` is `PAYLOAD_MULTIPLIER * i`.
    const PAYLOAD_MULTIPLIER: usize = 113;

    let temp = TestDir::temp();
    let cache = Cache::new(temp.path("."), 3, 9);

    for i in 0..200 {
        let name = format!("{}", i);

        let temp_dir = cache.temp_dir(None).expect("temp_dir must succeed");
        let tmp = NamedTempFile::new_in(temp_dir).expect("new temp file must succeed");
        std::fs::write(tmp.path(), format!("{}", PAYLOAD_MULTIPLIER * i))
            .expect("write must succeed");
        // It shouldn't matter if we PUT or SET.
        if (i % 2) != 0 {
            cache
                .put(Key::new(&name, i as u64, i as u64 + 42), tmp.path())
                .expect("put must succeed");
        } else {
            cache
                .set(Key::new(&name, i as u64, i as u64 + 42), tmp.path())
                .expect("set must succeed");
        }
    }

    let present: usize = (0..200)
        .map(|i| {
            let name = format!("{}", i);
            match cache
                .get(Key::new(&name, i as u64, i as u64 + 42))
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
    assert!(present <= 18);
}

/// Publish a file, make sure we can read it, then overwrite, and
/// confirm that the new contents are visible.
#[test]
fn test_set() {
    use std::io::{Read, Write};
    use tempfile::NamedTempFile;
    use test_dir::{DirBuilder, TestDir};

    let temp = TestDir::temp();
    let cache = Cache::new(temp.path("."), 0, 0);

    {
        let tmp = NamedTempFile::new_in(cache.temp_dir(None).expect("temp_dir must succeed"))
            .expect("new temp file must succeed");
        tmp.as_file().write_all(b"v1").expect("write must succeed");

        cache
            .set(Key::new("entry", 1, 2), tmp.path())
            .expect("initial set must succeed");
    }

    {
        let mut cached = cache
            .get(Key::new("entry", 1, 2))
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
            .set(Key::new("entry", 1, 2), tmp.path())
            .expect("overwrite must succeed");
    }

    {
        let mut cached = cache
            .get(Key::new("entry", 1, 2))
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
    let cache = Cache::new(temp.path("."), 0, 0);

    {
        let tmp = NamedTempFile::new_in(cache.temp_dir(None).expect("temp_dir must succeed"))
            .expect("new temp file must succeed");
        tmp.as_file().write_all(b"v1").expect("write must succeed");

        cache
            .set(Key::new("entry", 1, 2), tmp.path())
            .expect("initial set must succeed");
    }

    // Now put; it should not take.
    {
        let tmp = NamedTempFile::new_in(cache.temp_dir(None).expect("temp_dir must succeed"))
            .expect("new temp file must succeed");
        tmp.as_file().write_all(b"v2").expect("write must succeed");

        cache
            .put(Key::new("entry", 1, 2), tmp.path())
            .expect("put must succeed");
    }

    {
        let mut cached = cache
            .get(Key::new("entry", 1, 2))
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
    let cache = Cache::new(temp.path("."), 2, 600);

    for i in 0..2000 {
        // After the first write, we should find our file.
        assert_eq!(
            cache
                .touch(Key::new("0", 0, 42))
                .expect("touch must succeed"),
            i > 0
        );

        let name = format!("{}", i);

        let temp_dir = cache.temp_dir(None).expect("temp_dir must succeed");
        let tmp = NamedTempFile::new_in(temp_dir).expect("new temp file must succeed");
        std::fs::write(tmp.path(), format!("{}", PAYLOAD_MULTIPLIER * i))
            .expect("write must succeed");
        cache
            .put(Key::new(&name, i as u64, i as u64 + 42), tmp.path())
            .expect("put must succeed");
        if i == 0 {
            // Make sure file "0" is measurably older than the others.
            std::thread::sleep(std::time::Duration::from_secs(2));
        }
    }

    let mut file = cache
        .get(Key::new("0", 0, 42))
        .expect("get must succeed")
        .expect("file must be found");
    let mut buf = Vec::new();
    file.read_to_end(&mut buf).expect("read must succeed");
    assert_eq!(buf, b"0");
}
