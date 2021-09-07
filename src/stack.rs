//! We expect most callers to interact with Kismet via the `Cache`
//! struct defined here.  A `Cache` hides the difference in behaviour
//! between plain and sharded caches via late binding, and lets
//! callers transparently handle misses by looking in a series of
//! secondary cache directories.
use std::borrow::Cow;
use std::fs::File;
use std::io::Error;
use std::io::ErrorKind;
use std::io::Result;
use std::path::Path;
use std::sync::Arc;
use tempfile::NamedTempFile;

use crate::Key;
use crate::PlainCache;
use crate::ReadOnlyCache;
use crate::ReadOnlyCacheBuilder;
use crate::ShardedCache;

/// The `FullCache` trait exposes both read and write operations as
/// implemented by sharded and plain caches.
trait FullCache:
    std::fmt::Debug + Sync + Send + std::panic::RefUnwindSafe + std::panic::UnwindSafe
{
    /// Returns a read-only file for `key` in the cache directory if
    /// it exists, or None if there is no such file.
    ///
    /// Implicitly "touches" the cached file if it exists.
    fn get(&self, key: Key) -> Result<Option<File>>;

    /// Returns a temporary directory suitable for temporary files
    /// that will be published as `key`.
    fn temp_dir(&self, key: Key) -> Result<Cow<Path>>;

    /// Inserts or overwrites the file at `value` as `key` in the
    /// sharded cache directory.
    ///
    /// Always consumes the file at `value` on success; may consume it
    /// on error.
    fn set(&self, key: Key, value: &Path) -> Result<()>;

    /// Inserts the file at `value` as `key` in the cache directory if
    /// there is no such cached entry already, or touches the cached
    /// file if it already exists.
    ///
    /// Always consumes the file at `value` on success; may consume it
    /// on error.
    fn put(&self, key: Key, value: &Path) -> Result<()>;

    /// Marks the cached file `key` as newly used, if it exists.
    ///
    /// Returns whether a file for `key` exists in the cache.
    fn touch(&self, key: Key) -> Result<bool>;
}

impl FullCache for PlainCache {
    fn get(&self, key: Key) -> Result<Option<File>> {
        PlainCache::get(self, key.name)
    }

    fn temp_dir(&self, _key: Key) -> Result<Cow<Path>> {
        PlainCache::temp_dir(self)
    }

    fn set(&self, key: Key, value: &Path) -> Result<()> {
        PlainCache::set(self, key.name, value)
    }

    fn put(&self, key: Key, value: &Path) -> Result<()> {
        PlainCache::put(self, key.name, value)
    }

    fn touch(&self, key: Key) -> Result<bool> {
        PlainCache::touch(self, key.name)
    }
}

impl FullCache for ShardedCache {
    fn get(&self, key: Key) -> Result<Option<File>> {
        ShardedCache::get(self, key)
    }

    fn temp_dir(&self, key: Key) -> Result<Cow<Path>> {
        ShardedCache::temp_dir(self, Some(key))
    }

    fn set(&self, key: Key, value: &Path) -> Result<()> {
        ShardedCache::set(self, key, value)
    }

    fn put(&self, key: Key, value: &Path) -> Result<()> {
        ShardedCache::put(self, key, value)
    }

    fn touch(&self, key: Key) -> Result<bool> {
        ShardedCache::touch(self, key)
    }
}

/// Construct a `Cache` with this builder.  The resulting cache will
/// always first access its write-side cache (if defined), and, on
/// misses, will attempt to service `get` and `touch` calls by
/// iterating over the read-only caches.
#[derive(Debug, Default)]
pub struct CacheBuilder {
    write_side: Option<Arc<dyn FullCache>>,
    read_side: ReadOnlyCacheBuilder,
}

/// A `Cache` wraps either up to one plain or sharded read-write cache
/// in a convenient interface, and may optionally fulfill read
/// operations by deferring to a list of read-only cache when the
/// read-write cache misses.
#[derive(Clone, Debug, Default)]
pub struct Cache {
    write_side: Option<Arc<dyn FullCache>>,
    read_side: ReadOnlyCache,
}

/// Where does a cache hit come from: the primary read-write cache, or
/// one of the secondary read-only caches?
pub enum CacheHit<'a> {
    /// The file was found in the primary read-write cache; promoting
    /// it is a no-op.
    Primary(&'a mut File),
    /// The file was found in one of the secondary read-only caches.
    /// Promoting it to the primary cache will require a full copy.
    Secondary(&'a mut File),
}

/// What to do with a cache hit in a `get_or_update` call?
pub enum CacheHitAction {
    /// Return the cache hit as is.
    Accept,
    /// Return the cache hit after promoting it to the current write
    /// cache directory, if necessary.
    Promote,
    /// Replace with and return a new file.
    Replace,
}

impl CacheBuilder {
    /// Returns a fresh empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the read-write cache directory to `path`.
    ///
    /// The read-write cache will be a plain cache directory if
    /// `num_shards <= 1`, and a sharded directory otherwise.
    pub fn writer(self, path: impl AsRef<Path>, num_shards: usize, total_capacity: usize) -> Self {
        if num_shards <= 1 {
            self.plain_writer(path, total_capacity)
        } else {
            self.sharded_writer(path, num_shards, total_capacity)
        }
    }

    /// Sets the read-write cache directory to a plain directory at
    /// `path`, with a target file count of up to `capacity`.
    pub fn plain_writer(mut self, path: impl AsRef<Path>, capacity: usize) -> Self {
        self.write_side.insert(Arc::new(PlainCache::new(
            path.as_ref().to_owned(),
            capacity,
        )));
        self
    }

    /// Sets the read-write cache directory to a sharded directory at
    /// `path`, with `num_shards` subdirectories and a target file
    /// count of up to `capacity` for the entire cache.
    pub fn sharded_writer(
        mut self,
        path: impl AsRef<Path>,
        num_shards: usize,
        total_capacity: usize,
    ) -> Self {
        self.write_side.insert(Arc::new(ShardedCache::new(
            path.as_ref().to_owned(),
            num_shards,
            total_capacity,
        )));
        self
    }

    /// Adds a new read-only cache directory at `path` to the end of the
    /// cache builder's search list.
    ///
    /// Adds a plain cache directory if `num_shards <= 1`, and a sharded
    /// directory otherwise.
    pub fn reader(mut self, path: impl AsRef<Path>, num_shards: usize) -> Self {
        self.read_side = self.read_side.cache(path, num_shards);
        self
    }

    /// Adds a new plain (unsharded) read-only cache directory at
    /// `path` to the end of the cache builder's search list.
    pub fn plain_reader(mut self, path: impl AsRef<Path>) -> Self {
        self.read_side = self.read_side.plain(path);
        self
    }

    /// Adds a new sharded read-only cache directory at `path` to the
    /// end of the cache builder's search list.
    pub fn sharded_reader(mut self, path: impl AsRef<Path>, num_shards: usize) -> Self {
        self.read_side = self.read_side.sharded(path, num_shards);
        self
    }

    /// Returns a fresh `Cache` for the builder's write cache and
    /// additional search list of read-only cache directories.
    pub fn build(self) -> Cache {
        Cache {
            write_side: self.write_side,
            read_side: self.read_side.build(),
        }
    }
}

impl Cache {
    /// Attempts to open a read-only file for `key`.  The `Cache` will
    /// query each its write cache (if any), followed by the list of
    /// additional read-only cache, in definition order, and return a
    /// read-only file for the first hit.
    ///
    /// Fails with `ErrorKind::InvalidInput` if `key.name` is invalid
    /// (empty, or starts with a dot or a forward or back slash).
    ///
    /// Returns `None` if no file for `key` can be found in any of the
    /// constituent caches, and bubbles up the first I/O error
    /// encountered, if any.
    pub fn get<'a>(&self, key: impl Into<Key<'a>>) -> Result<Option<File>> {
        fn doit(
            write_side: Option<&dyn FullCache>,
            read_side: &ReadOnlyCache,
            key: Key,
        ) -> Result<Option<File>> {
            if let Some(write) = write_side {
                if let Some(ret) = write.get(key)? {
                    return Ok(Some(ret));
                }
            }

            read_side.get(key)
        }

        doit(
            self.write_side.as_ref().map(AsRef::as_ref),
            &self.read_side,
            key.into(),
        )
    }

    /// Attempts to find a cache entry for `key`.  If there is none,
    /// populates the cache with a file filled by `populate`.  Returns
    /// a file in all cases (unless the call fails with an error).
    ///
    /// Fails with `ErrorKind::InvalidInput` if `key.name` is invalid
    /// (empty, or starts with a dot or a forward or back slash).
    pub fn ensure<'a>(
        &self,
        key: impl Into<Key<'a>>,
        populate: impl FnOnce(&mut File) -> Result<()>,
    ) -> Result<File> {
        fn judge(_: CacheHit) -> CacheHitAction {
            CacheHitAction::Promote
        }

        self.get_or_update(key, judge, |dst, _| populate(dst))
    }

    /// Attempts to find a cache entry for `key`.  If there is none,
    /// populates the write cache (if possible) with a file, once
    /// filled by `populate`; otherwise obeys the value returned by
    /// `judge` to determine what to do with the hit.
    ///
    /// Fails with `ErrorKind::InvalidInput` if `key.name` is invalid
    /// (empty, or starts with a dot or a forward or back slash).
    ///
    /// When we need to populate a new file, `populate` is called with
    /// a mutable reference to the destination file, and the old
    /// cached file (in whatever state `judge` left it), if available.
    pub fn get_or_update<'a>(
        &self,
        key: impl Into<Key<'a>>,
        judge: impl FnOnce(CacheHit) -> CacheHitAction,
        populate: impl FnOnce(&mut File, Option<File>) -> Result<()>,
    ) -> Result<File> {
        // Attempts to return the `FullCache` for this `Cache`.
        fn get_write_cache(this: &Cache) -> Result<&dyn FullCache> {
            match this.write_side.as_ref() {
                Some(cache) => Ok(cache.as_ref()),
                None => Err(Error::new(
                    ErrorKind::Unsupported,
                    "no kismet write cache defined",
                )),
            }
        }

        // Promotes `file` to `cache`.
        fn promote(cache: &dyn FullCache, key: Key, mut file: File) -> Result<File> {
            use std::io::Seek;

            let mut tmp = NamedTempFile::new_in(cache.temp_dir(key)?)?;
            std::io::copy(&mut file, tmp.as_file_mut())?;

            // Force the destination file's contents to disk before
            // adding it to the read-write cache: the caller can't
            // tell us whether they want a `fsync`, so let's be safe
            // and assume they do.
            tmp.as_file().sync_all()?;
            cache.put(key, tmp.path())?;

            // We got a read-only file.  Rewind it before returning.
            file.seek(std::io::SeekFrom::Start(0))?;
            Ok(file)
        }

        let cache = get_write_cache(self)?;
        let key: Key = key.into();

        // Overwritten with `Some(file)` when replacing `file`.
        let mut old = None;
        if let Some(mut file) = cache.get(key)? {
            match judge(CacheHit::Primary(&mut file)) {
                // Promote is a no-op if the file is already in the write cache.
                CacheHitAction::Accept | CacheHitAction::Promote => return Ok(file),
                CacheHitAction::Replace => old = Some(file),
            }
        } else if let Some(mut file) = self.read_side.get(key)? {
            match judge(CacheHit::Secondary(&mut file)) {
                CacheHitAction::Accept => return Ok(file),
                CacheHitAction::Promote => return promote(get_write_cache(self)?, key, file),
                CacheHitAction::Replace => old = Some(file),
            }
        }

        let replace = old.is_some();
        // We either have to replace or ensure there is a cache entry.
        // Either way, start by populating a temporary file.
        let mut tmp = NamedTempFile::new_in(cache.temp_dir(key)?)?;
        populate(tmp.as_file_mut(), old)?;

        // Grab a read-only return value before publishing the file.
        let path = tmp.path();
        let ret = File::open(path)?;
        if replace {
            cache.set(key, path)?;
        } else {
            cache.put(key, path)?;
        }

        Ok(ret)
    }

    /// Inserts or overwrites the file at `value` as `key` in the
    /// write cache directory.  This will always fail with
    /// `Unsupported` if no write cache was defined.
    ///
    /// Fails with `ErrorKind::InvalidInput` if `key.name` is invalid
    /// (empty, or starts with a dot or a forward or back slash).
    ///
    /// Always consumes the file at `value` on success; may consume it
    /// on error.
    pub fn set<'a>(&self, key: impl Into<Key<'a>>, value: impl AsRef<Path>) -> Result<()> {
        match self.write_side.as_ref() {
            Some(write) => write.set(key.into(), value.as_ref()),
            None => Err(Error::new(
                ErrorKind::Unsupported,
                "no kismet write cache defined",
            )),
        }
    }

    /// Inserts the file at `value` as `key` in the cache directory if
    /// there is no such cached entry already, or touches the cached
    /// file if it already exists.
    ///
    /// Fails with `ErrorKind::InvalidInput` if `key.name` is invalid
    /// (empty, or starts with a dot or a forward or back slash).
    ///
    /// Always consumes the file at `value` on success; may consume it
    /// on error.
    pub fn put<'a>(&self, key: impl Into<Key<'a>>, value: impl AsRef<Path>) -> Result<()> {
        match self.write_side.as_ref() {
            Some(write) => write.put(key.into(), value.as_ref()),
            None => Err(Error::new(
                ErrorKind::Unsupported,
                "no kismet write cache defined",
            )),
        }
    }

    /// Marks a cache entry for `key` as accessed (read).  The `Cache`
    /// will touch the same file that would be returned by `get`.
    ///
    /// Fails with `ErrorKind::InvalidInput` if `key.name` is invalid
    /// (empty, or starts with a dot or a forward or back slash).
    ///
    /// Returns whether a file for `key` could be found, and bubbles
    /// up the first I/O error encountered, if any.
    pub fn touch<'a>(&self, key: impl Into<Key<'a>>) -> Result<bool> {
        fn doit(
            write_side: Option<&dyn FullCache>,
            read_side: &ReadOnlyCache,
            key: Key,
        ) -> Result<bool> {
            if let Some(write) = write_side {
                if write.touch(key)? {
                    return Ok(true);
                }
            }

            read_side.touch(key)
        }

        doit(
            self.write_side.as_ref().map(AsRef::as_ref),
            &self.read_side,
            key.into(),
        )
    }
}

#[cfg(test)]
mod test {
    use std::io::ErrorKind;

    use crate::Cache;
    use crate::CacheBuilder;
    use crate::CacheHit;
    use crate::CacheHitAction;
    use crate::Key;
    use crate::PlainCache;
    use crate::ShardedCache;

    struct TestKey {
        key: String,
    }

    impl TestKey {
        fn new(key: &str) -> TestKey {
            TestKey {
                key: key.to_string(),
            }
        }
    }

    impl<'a> From<&'a TestKey> for Key<'a> {
        fn from(x: &'a TestKey) -> Key<'a> {
            Key::new(&x.key, 0, 1)
        }
    }

    // No cache defined -> read calls should successfully do nothing,
    // write calls should fail.
    #[test]
    fn empty() {
        let cache: Cache = Default::default();

        assert!(matches!(cache.get(&TestKey::new("foo")), Ok(None)));
        assert!(
            matches!(cache.ensure(&TestKey::new("foo"), |_| unreachable!("should not be called when there is no write side")),
                         Err(e) if e.kind() == ErrorKind::Unsupported)
        );
        assert!(matches!(cache.set(&TestKey::new("foo"), "/tmp/foo"),
                         Err(e) if e.kind() == ErrorKind::Unsupported));
        assert!(matches!(cache.put(&TestKey::new("foo"), "/tmp/foo"),
                         Err(e) if e.kind() == ErrorKind::Unsupported));
        assert!(matches!(cache.touch(&TestKey::new("foo")), Ok(false)));
    }

    // Fail to find a file, ensure it, then see that we can get it.
    #[test]
    fn test_ensure() {
        use std::io::{Read, Write};
        use test_dir::{DirBuilder, TestDir};

        let temp = TestDir::temp();
        let cache = CacheBuilder::new().writer(temp.path("."), 1, 10).build();
        let key = TestKey::new("foo");

        // The file doesn't exist initially.
        assert!(matches!(cache.get(&key), Ok(None)));

        {
            let mut populated = cache
                .ensure(&key, |file| file.write_all(b"test"))
                .expect("ensure must succeed");

            let mut dst = Vec::new();
            populated.read_to_end(&mut dst).expect("read must succeed");
            assert_eq!(&dst, b"test");
        }

        // And now get the file again.
        {
            let mut fetched = cache
                .get(&key)
                .expect("get must succeed")
                .expect("file must be found");

            let mut dst = Vec::new();
            fetched.read_to_end(&mut dst).expect("read must succeed");
            assert_eq!(&dst, b"test");
        }

        // And make sure a later `ensure` call just grabs the file.
        {
            let mut populated = cache
                .ensure(&key, |_| {
                    unreachable!("should not be called for an extant file")
                })
                .expect("ensure must succeed");

            let mut dst = Vec::new();
            populated.read_to_end(&mut dst).expect("read must succeed");
            assert_eq!(&dst, b"test");
        }
    }

    // Use a two-level cache, and make sure `ensure` promotes copies from
    // the backup to the primary location.
    #[test]
    fn test_ensure_promote() {
        use std::io::{Read, Write};
        use tempfile::NamedTempFile;
        use test_dir::{DirBuilder, FileType, TestDir};

        let temp = TestDir::temp()
            .create("cache", FileType::Dir)
            .create("extra_plain", FileType::Dir);

        // Populate the plain cache in `extra_plain` with one file.
        {
            let cache = PlainCache::new(temp.path("extra_plain"), 10);

            let tmp = NamedTempFile::new_in(cache.temp_dir().expect("temp_dir must succeed"))
                .expect("new temp file must succeed");
            tmp.as_file()
                .write_all(b"initial")
                .expect("write must succeed");

            cache.put("foo", tmp.path()).expect("put must succeed");
        }

        let cache = CacheBuilder::new()
            .writer(temp.path("cache"), 1, 10)
            .plain_reader(temp.path("extra_plain"))
            .build();
        let key = TestKey::new("foo");

        // The file is found initially.
        {
            let mut fetched = cache
                .get(&key)
                .expect("get must succeed")
                .expect("file must be found");

            let mut dst = Vec::new();
            fetched.read_to_end(&mut dst).expect("read must succeed");
            assert_eq!(&dst, b"initial");
        }

        {
            let mut populated = cache
                .ensure(&key, |_| {
                    unreachable!("should not be called for an extant file")
                })
                .expect("ensure must succeed");

            let mut dst = Vec::new();
            populated.read_to_end(&mut dst).expect("read must succeed");
            assert_eq!(&dst, b"initial");
        }

        // And now get the file again, and make sure it doesn't come from the
        // backup location.
        {
            let new_cache = CacheBuilder::new()
                .writer(temp.path("cache"), 1, 10)
                .build();
            let mut fetched = new_cache
                .get(&key)
                .expect("get must succeed")
                .expect("file must be found");

            let mut dst = Vec::new();
            fetched.read_to_end(&mut dst).expect("read must succeed");
            assert_eq!(&dst, b"initial");
        }
    }

    // Use a two-level cache, get_or_update with an `Accept` judgement.
    // We should leave everything where it is.
    #[test]
    fn test_get_or_update_accept() {
        use std::io::{Read, Write};
        use tempfile::NamedTempFile;
        use test_dir::{DirBuilder, FileType, TestDir};

        let temp = TestDir::temp()
            .create("cache", FileType::Dir)
            .create("extra_plain", FileType::Dir);

        // Populate the plain cache in `extra_plain` with one file.
        {
            let cache = PlainCache::new(temp.path("extra_plain"), 10);

            let tmp = NamedTempFile::new_in(cache.temp_dir().expect("temp_dir must succeed"))
                .expect("new temp file must succeed");
            tmp.as_file()
                .write_all(b"initial")
                .expect("write must succeed");

            cache.put("foo", tmp.path()).expect("put must succeed");
        }

        let cache = CacheBuilder::new()
            // Make it sharded, because why not?
            .writer(temp.path("cache"), 2, 10)
            .plain_reader(temp.path("extra_plain"))
            .build();
        let key = TestKey::new("foo");
        let key2 = TestKey::new("bar");

        // The file is found initially, in the backup cache.
        {
            let mut fetched = cache
                .get_or_update(
                    &key,
                    |hit| {
                        assert!(matches!(hit, CacheHit::Secondary(_)));
                        CacheHitAction::Accept
                    },
                    |_, _| unreachable!("should not have to fill an extant file"),
                )
                .expect("get_or_update must succeed");

            let mut dst = Vec::new();
            fetched.read_to_end(&mut dst).expect("read must succeed");
            assert_eq!(&dst, b"initial");
        }

        // Let's try again with a file that does not exist yet.
        {
            let mut fetched = cache
                .get_or_update(
                    &key2,
                    |_| unreachable!("should not be called"),
                    |file, old| {
                        assert!(old.is_none());
                        file.write_all(b"updated")
                    },
                )
                .expect("get_or_update must succeed");

            let mut dst = Vec::new();
            fetched.read_to_end(&mut dst).expect("read must succeed");
            assert_eq!(&dst, b"updated");
        }

        // The new file is now found.
        {
            let mut fetched = cache
                .get_or_update(
                    &key2,
                    |hit| {
                        assert!(matches!(hit, CacheHit::Primary(_)));
                        CacheHitAction::Accept
                    },
                    |_, _| unreachable!("should not have to fill an extant file"),
                )
                .expect("get_or_update must succeed");

            let mut dst = Vec::new();
            fetched.read_to_end(&mut dst).expect("read must succeed");
            assert_eq!(&dst, b"updated");
        }

        // And now get the files again, and make sure they don't
        // come from the backup location.
        {
            let new_cache = CacheBuilder::new()
                .writer(temp.path("cache"), 2, 10)
                .build();

            // The new cache shouldn't have the old key.
            assert!(matches!(new_cache.touch(&key), Ok(false)));

            // But it should have `key2`.
            let mut fetched = new_cache
                .get(&key2)
                .expect("get must succeed")
                .expect("file must be found");

            let mut dst = Vec::new();
            fetched.read_to_end(&mut dst).expect("read must succeed");
            assert_eq!(&dst, b"updated");
        }
    }

    // Use a two-level cache, get_or_update with a `Replace` judgement.
    // We should always overwrite everything to the write cache.
    #[test]
    fn test_get_or_update_replace() {
        use std::io::{Read, Write};
        use tempfile::NamedTempFile;
        use test_dir::{DirBuilder, FileType, TestDir};

        let temp = TestDir::temp()
            .create("cache", FileType::Dir)
            .create("extra_plain", FileType::Dir);

        // Populate the plain cache in `extra_plain` with one file.
        {
            let cache = PlainCache::new(temp.path("extra_plain"), 10);

            let tmp = NamedTempFile::new_in(cache.temp_dir().expect("temp_dir must succeed"))
                .expect("new temp file must succeed");
            tmp.as_file()
                .write_all(b"initial")
                .expect("write must succeed");

            cache.put("foo", tmp.path()).expect("put must succeed");
        }

        let cache = CacheBuilder::new()
            // Make it sharded, because why not?
            .writer(temp.path("cache"), 2, 10)
            .plain_reader(temp.path("extra_plain"))
            .build();
        let key = TestKey::new("foo");

        {
            let mut fetched = cache
                .get_or_update(
                    &key,
                    |hit| {
                        assert!(matches!(hit, CacheHit::Secondary(_)));
                        CacheHitAction::Replace
                    },
                    |file, old| {
                        // Make sure the `old` file is the "initial" file.
                        let mut prev = old.expect("must have old data");
                        let mut dst = Vec::new();
                        prev.read_to_end(&mut dst).expect("read must succeed");
                        assert_eq!(&dst, b"initial");

                        file.write_all(b"replace1")
                    },
                )
                .expect("get_or_update must succeed");

            let mut dst = Vec::new();
            fetched.read_to_end(&mut dst).expect("read must succeed");
            assert_eq!(&dst, b"replace1");
        }

        // Re-read the file.
        {
            let mut fetched = cache
                .get(&key)
                .expect("get must succeed")
                .expect("file should be found");

            let mut dst = Vec::new();
            fetched.read_to_end(&mut dst).expect("read must succeed");
            assert_eq!(&dst, b"replace1");
        }

        // Update it again.
        {
            let mut fetched = cache
                .get_or_update(
                    &key,
                    |hit| {
                        assert!(matches!(hit, CacheHit::Primary(_)));
                        CacheHitAction::Replace
                    },
                    |file, old| {
                        // Make sure the `old` file is the "initial" file.
                        let mut prev = old.expect("must have old data");
                        let mut dst = Vec::new();
                        prev.read_to_end(&mut dst).expect("read must succeed");
                        assert_eq!(&dst, b"replace1");

                        file.write_all(b"replace2")
                    },
                )
                .expect("get_or_update must succeed");

            let mut dst = Vec::new();
            fetched.read_to_end(&mut dst).expect("read must succeed");
            assert_eq!(&dst, b"replace2");
        }

        // The new file is now found.
        {
            let mut fetched = cache
                .get_or_update(
                    &key,
                    |hit| {
                        assert!(matches!(hit, CacheHit::Primary(_)));
                        CacheHitAction::Replace
                    },
                    |file, old| {
                        // Make sure the `old` file is the "replace2" file.
                        let mut prev = old.expect("must have old data");
                        let mut dst = Vec::new();
                        prev.read_to_end(&mut dst).expect("read must succeed");
                        assert_eq!(&dst, b"replace2");

                        file.write_all(b"replace3")
                    },
                )
                .expect("get_or_update must succeed");

            let mut dst = Vec::new();
            fetched.read_to_end(&mut dst).expect("read must succeed");
            assert_eq!(&dst, b"replace3");
        }

        // And now get the same file again, and make sure it doesn't
        // come from the backup location.
        {
            let new_cache = CacheBuilder::new()
                .writer(temp.path("cache"), 2, 10)
                .build();

            // But it should have `key2`.
            let mut fetched = new_cache
                .get(&key)
                .expect("get must succeed")
                .expect("file must be found");

            let mut dst = Vec::new();
            fetched.read_to_end(&mut dst).expect("read must succeed");
            assert_eq!(&dst, b"replace3");
        }
    }

    // Smoke test a wrapped plain cache.
    #[test]
    fn smoke_test_plain() {
        use std::io::{Read, Write};
        use tempfile::NamedTempFile;
        use test_dir::{DirBuilder, FileType, TestDir};

        let temp = TestDir::temp()
            .create("cache", FileType::Dir)
            .create("extra", FileType::Dir);

        // Populate the plain cache in `extra` with two files, "b" and "c".
        {
            let cache = PlainCache::new(temp.path("extra"), 10);

            let tmp = NamedTempFile::new_in(cache.temp_dir().expect("temp_dir must succeed"))
                .expect("new temp file must succeed");
            tmp.as_file()
                .write_all(b"extra")
                .expect("write must succeed");

            cache.put("b", tmp.path()).expect("put must succeed");

            let tmp2 = NamedTempFile::new_in(cache.temp_dir().expect("temp_dir must succeed"))
                .expect("new temp file must succeed");
            tmp2.as_file()
                .write_all(b"extra2")
                .expect("write must succeed");

            cache.put("c", tmp2.path()).expect("put must succeed");
        }

        let cache = CacheBuilder::new()
            .writer(temp.path("cache"), 1, 10)
            .reader(temp.path("extra"), 1)
            .build();

        // There shouldn't be anything for "a"
        assert!(matches!(cache.get(&TestKey::new("a")), Ok(None)));
        assert!(matches!(cache.touch(&TestKey::new("a")), Ok(false)));

        // We should be able to touch "b"
        assert!(matches!(cache.touch(&TestKey::new("b")), Ok(true)));

        // And its contents should match that of the "extra" cache dir.
        {
            let mut b_file = cache
                .get(&TestKey::new("b"))
                .expect("must succeed")
                .expect("must exist");
            let mut dst = Vec::new();
            b_file.read_to_end(&mut dst).expect("read must succeed");
            assert_eq!(&dst, b"extra");
        }

        // Now populate "a" and "b" in the cache.
        {
            let tmp = NamedTempFile::new_in(temp.path(".")).expect("new temp file must succeed");

            tmp.as_file()
                .write_all(b"write")
                .expect("write must succeed");
            cache
                .put(&TestKey::new("a"), tmp.path())
                .expect("put must succeed");
        }

        {
            let tmp = NamedTempFile::new_in(temp.path(".")).expect("new temp file must succeed");

            tmp.as_file()
                .write_all(b"write2")
                .expect("write must succeed");
            cache
                .put(&TestKey::new("b"), tmp.path())
                .expect("put must succeed");
        }

        // And overwrite "a"
        {
            let tmp = NamedTempFile::new_in(temp.path(".")).expect("new temp file must succeed");

            tmp.as_file()
                .write_all(b"write3")
                .expect("write must succeed");
            cache
                .set(&TestKey::new("a"), tmp.path())
                .expect("set must succeed");
        }

        // We should find:
        // a => write3
        // b => write2
        // c => extra2

        // So we should be able to touch everything.
        assert!(matches!(cache.touch(&TestKey::new("a")), Ok(true)));
        assert!(matches!(cache.touch(&TestKey::new("b")), Ok(true)));
        assert!(matches!(cache.touch(&TestKey::new("c")), Ok(true)));

        // And read the expected contents.
        {
            let mut a_file = cache
                .get(&TestKey::new("a"))
                .expect("must succeed")
                .expect("must exist");
            let mut dst = Vec::new();
            a_file.read_to_end(&mut dst).expect("read must succeed");
            assert_eq!(&dst, b"write3");
        }

        {
            let mut b_file = cache
                .get(&TestKey::new("b"))
                .expect("must succeed")
                .expect("must exist");
            let mut dst = Vec::new();
            b_file.read_to_end(&mut dst).expect("read must succeed");
            assert_eq!(&dst, b"write2");
        }

        {
            let mut c_file = cache
                .get(&TestKey::new("c"))
                .expect("must succeed")
                .expect("must exist");
            let mut dst = Vec::new();
            c_file.read_to_end(&mut dst).expect("read must succeed");
            assert_eq!(&dst, b"extra2");
        }
    }

    // Smoke test a wrapped sharded cache.
    #[test]
    fn smoke_test_sharded() {
        use std::io::{Read, Write};
        use tempfile::NamedTempFile;
        use test_dir::{DirBuilder, FileType, TestDir};

        let temp = TestDir::temp()
            .create("cache", FileType::Dir)
            .create("extra_plain", FileType::Dir)
            .create("extra_sharded", FileType::Dir);

        // Populate the plain cache in `extra_plain` with one file, "b".
        {
            let cache = PlainCache::new(temp.path("extra_plain"), 10);

            let tmp = NamedTempFile::new_in(cache.temp_dir().expect("temp_dir must succeed"))
                .expect("new temp file must succeed");
            tmp.as_file()
                .write_all(b"extra_plain")
                .expect("write must succeed");

            cache.put("b", tmp.path()).expect("put must succeed");
        }

        // And now add "c" in the sharded `extra_sharded` cache.
        {
            let cache = ShardedCache::new(temp.path("extra_sharded"), 10, 10);

            let tmp = NamedTempFile::new_in(cache.temp_dir(None).expect("temp_dir must succeed"))
                .expect("new temp file must succeed");
            tmp.as_file()
                .write_all(b"extra_sharded")
                .expect("write must succeed");

            cache
                .put((&TestKey::new("c")).into(), tmp.path())
                .expect("put must succeed");
        }

        let cache = CacheBuilder::new()
            .plain_writer(temp.path("cache"), 10)
            // Override the writer with a sharded cache
            .writer(temp.path("cache"), 10, 10)
            .plain_reader(temp.path("extra_plain"))
            .sharded_reader(temp.path("extra_sharded"), 10)
            .build();

        // There shouldn't be anything for "a"
        assert!(matches!(cache.get(&TestKey::new("a")), Ok(None)));
        assert!(matches!(cache.touch(&TestKey::new("a")), Ok(false)));

        // We should be able to touch "b"
        assert!(matches!(cache.touch(&TestKey::new("b")), Ok(true)));

        // And its contents should match that of the "extra" cache dir.
        {
            let mut b_file = cache
                .get(&TestKey::new("b"))
                .expect("must succeed")
                .expect("must exist");
            let mut dst = Vec::new();
            b_file.read_to_end(&mut dst).expect("read must succeed");
            assert_eq!(&dst, b"extra_plain");
        }

        // Similarly for "c"
        {
            let mut c_file = cache
                .get(&TestKey::new("c"))
                .expect("must succeed")
                .expect("must exist");
            let mut dst = Vec::new();
            c_file.read_to_end(&mut dst).expect("read must succeed");
            assert_eq!(&dst, b"extra_sharded");
        }

        // Now populate "a" and "b" in the cache.
        {
            let tmp = NamedTempFile::new_in(temp.path(".")).expect("new temp file must succeed");

            tmp.as_file()
                .write_all(b"write")
                .expect("write must succeed");
            cache
                .set(&TestKey::new("a"), tmp.path())
                .expect("set must succeed");
        }

        {
            let tmp = NamedTempFile::new_in(temp.path(".")).expect("new temp file must succeed");

            tmp.as_file()
                .write_all(b"write2")
                .expect("write must succeed");
            cache
                .set(&TestKey::new("b"), tmp.path())
                .expect("set must succeed");
        }

        // And fail to update "a" with a put.
        {
            let tmp = NamedTempFile::new_in(temp.path(".")).expect("new temp file must succeed");

            tmp.as_file()
                .write_all(b"write3")
                .expect("write must succeed");
            cache
                .put(&TestKey::new("a"), tmp.path())
                .expect("put must succeed");
        }

        // We should find:
        // a => write
        // b => write2
        // c => extra_sharded

        // So we should be able to touch everything.
        assert!(matches!(cache.touch(&TestKey::new("a")), Ok(true)));
        assert!(matches!(cache.touch(&TestKey::new("b")), Ok(true)));
        assert!(matches!(cache.touch(&TestKey::new("c")), Ok(true)));

        // And read the expected contents.
        {
            let mut a_file = cache
                .get(&TestKey::new("a"))
                .expect("must succeed")
                .expect("must exist");
            let mut dst = Vec::new();
            a_file.read_to_end(&mut dst).expect("read must succeed");
            assert_eq!(&dst, b"write");
        }

        {
            let mut b_file = cache
                .get(&TestKey::new("b"))
                .expect("must succeed")
                .expect("must exist");
            let mut dst = Vec::new();
            b_file.read_to_end(&mut dst).expect("read must succeed");
            assert_eq!(&dst, b"write2");
        }

        {
            let mut c_file = cache
                .get(&TestKey::new("c"))
                .expect("must succeed")
                .expect("must exist");
            let mut dst = Vec::new();
            c_file.read_to_end(&mut dst).expect("read must succeed");
            assert_eq!(&dst, b"extra_sharded");
        }
    }
}
