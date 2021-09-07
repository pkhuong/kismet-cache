//! We expect most callers to interact with Kismet via the `Cache`
//! struct defined here.  A `Cache` hides the difference in behaviour
//! between plain and sharded caches via late binding, and lets
//! callers transparently handle misses by looking in a series of
//! secondary cache directories.
use std::fs::File;
use std::io::Error;
use std::io::ErrorKind;
use std::io::Result;
use std::path::Path;
use std::sync::Arc;

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

    /// Inserts or overwrites the file at `value` as `key` in the
    /// write cache directory.  This will always fail with
    /// `Unsupported` if no write cache was defined.
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
        assert!(matches!(cache.set(&TestKey::new("foo"), "/tmp/foo"),
                         Err(e) if e.kind() == ErrorKind::Unsupported));
        assert!(matches!(cache.put(&TestKey::new("foo"), "/tmp/foo"),
                         Err(e) if e.kind() == ErrorKind::Unsupported));
        assert!(matches!(cache.touch(&TestKey::new("foo")), Ok(false)));
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
