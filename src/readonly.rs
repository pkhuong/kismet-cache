//! A `ReadOnlyCache` wraps an arbitrary number of caches, and
//! attempts to satisfy `get` and `touch` requests by hitting each
//! cache in order.  For read-only usage, this should be a simple
//! and easy-to-use interface that erases the difference between plain
//! and sharded caches.
use std::fs::File;
#[allow(unused_imports)] // We refer to this enum in comments.
use std::io::ErrorKind;
use std::io::Result;
use std::path::Path;
use std::sync::Arc;

use crate::plain::Cache as PlainCache;
use crate::sharded::Cache as ShardedCache;
use crate::Key;

/// The `ReadSide` trait offers `get` and `touch`, as implemented by
/// both plain and sharded caches.
trait ReadSide:
    std::fmt::Debug + Sync + Send + std::panic::RefUnwindSafe + std::panic::UnwindSafe
{
    /// Returns a read-only file for `key` in the cache directory if
    /// it exists, or None if there is no such file.
    ///
    /// Implicitly "touches" the cached file if it exists.
    fn get(&self, key: Key) -> Result<Option<File>>;

    /// Marks the cached file `key` as newly used, if it exists.
    ///
    /// Returns whether a file for `key` exists in the cache.
    fn touch(&self, key: Key) -> Result<bool>;
}

impl ReadSide for PlainCache {
    fn get(&self, key: Key) -> Result<Option<File>> {
        PlainCache::get(self, key.name)
    }

    fn touch(&self, key: Key) -> Result<bool> {
        PlainCache::touch(self, key.name)
    }
}

impl ReadSide for ShardedCache {
    fn get(&self, key: Key) -> Result<Option<File>> {
        ShardedCache::get(self, key)
    }

    fn touch(&self, key: Key) -> Result<bool> {
        ShardedCache::touch(self, key)
    }
}

/// Construct a [`ReadOnlyCache`] with this builder.  The resulting
/// cache will access each constituent cache directory in the order
/// they were added.
///
/// The default builder is a fresh builder with no constituent cache.
#[derive(Debug, Default)]
pub struct ReadOnlyCacheBuilder {
    stack: Vec<Box<dyn ReadSide>>,
}

/// A [`ReadOnlyCache`] wraps an arbitrary number of
/// [`crate::plain::Cache`] and [`crate::sharded::Cache`], and attempts
/// to satisfy [`ReadOnlyCache::get`] and [`ReadOnlyCache::touch`]
/// requests by hitting each constituent cache in order.  This
/// interface hides the difference between plain and sharded cache
/// directories, and should be the first resort for read-only uses.
///
/// The default cache wraps an empty set of constituent caches.
///
/// [`ReadOnlyCache`] objects are stateless and cheap to clone; don't
/// put an [`Arc`] on them.  Avoid creating multiple
/// [`ReadOnlyCache`]s for the same stack of directories: there is no
/// internal state to maintain, so multiple instances simply waste
/// memory without any benefit.
#[derive(Clone, Debug)]
pub struct ReadOnlyCache {
    stack: Arc<[Box<dyn ReadSide>]>,
}

impl ReadOnlyCacheBuilder {
    /// Returns a fresh empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a new cache directory at `path` to the end of the cache
    /// builder's search list.
    ///
    /// Adds a plain cache directory if `num_shards <= 1`, and an
    /// actual sharded directory otherwise.
    pub fn cache(self, path: impl AsRef<Path>, num_shards: usize) -> Self {
        if num_shards <= 1 {
            self.plain(path)
        } else {
            self.sharded(path, num_shards)
        }
    }

    /// Adds a new plain cache directory at `path` to the end of the
    /// cache builder's search list.  A plain cache directory is
    /// merely a directory of files where the files' names match their
    /// key's name.
    pub fn plain(mut self, path: impl AsRef<Path>) -> Self {
        self.stack.push(Box::new(PlainCache::new(
            path.as_ref().to_owned(),
            usize::MAX,
        )));

        self
    }

    /// Adds a new sharded cache directory at `path` to the end of the
    /// cache builder's search list.
    pub fn sharded(mut self, path: impl AsRef<Path>, num_shards: usize) -> Self {
        self.stack.push(Box::new(ShardedCache::new(
            path.as_ref().to_owned(),
            num_shards,
            usize::MAX,
        )));
        self
    }

    /// Returns a fresh [`ReadOnlyCache`] for the builder's search list
    /// of constituent cache directories.
    pub fn build(self) -> ReadOnlyCache {
        ReadOnlyCache::new(self.stack)
    }
}

impl Default for ReadOnlyCache {
    fn default() -> ReadOnlyCache {
        ReadOnlyCache::new(Default::default())
    }
}

impl ReadOnlyCache {
    fn new(stack: Vec<Box<dyn ReadSide>>) -> ReadOnlyCache {
        ReadOnlyCache {
            stack: stack.into_boxed_slice().into(),
        }
    }

    /// Attempts to open a read-only file for `key`.  The
    /// [`ReadOnlyCache`] will query each constituent cache in order
    /// of registration, and return a read-only file for the first
    /// hit.
    ///
    /// Fails with [`ErrorKind::InvalidInput`] if `key.name` is
    /// invalid (empty, or starts with a dot or a forward or back slash).
    ///
    /// Returns [`None`] if no file for `key` can be found in any of
    /// the constituent caches, and bubbles up the first I/O error
    /// encountered, if any.
    ///
    /// In the worst case, each call to `get` attempts to open two
    /// files for each cache directory in the `ReadOnlyCache` stack.
    pub fn get<'a>(&self, key: impl Into<Key<'a>>) -> Result<Option<File>> {
        fn doit(stack: &[Box<dyn ReadSide>], key: Key) -> Result<Option<File>> {
            for cache in stack.iter() {
                if let Some(ret) = cache.get(key)? {
                    return Ok(Some(ret));
                }
            }

            Ok(None)
        }

        if self.stack.is_empty() {
            return Ok(None);
        }

        doit(&*self.stack, key.into())
    }

    /// Marks a cache entry for `key` as accessed (read).  The
    /// [`ReadOnlyCache`] will touch the same file that would be
    /// returned by `get`.
    ///
    /// Fails with [`ErrorKind::InvalidInput`] if `key.name` is
    /// invalid (empty, or starts with a dot or a forward or back slash).
    ///
    /// Returns whether a file for `key` could be found, and bubbles
    /// up the first I/O error encountered, if any.
    ///
    /// In the worst case, each call to `touch` attempts to update the
    /// access time on two files for each cache directory in the
    /// `ReadOnlyCache` stack.
    pub fn touch<'a>(&self, key: impl Into<Key<'a>>) -> Result<bool> {
        fn doit(stack: &[Box<dyn ReadSide>], key: Key) -> Result<bool> {
            for cache in stack.iter() {
                if cache.touch(key)? {
                    return Ok(true);
                }
            }

            Ok(false)
        }

        if self.stack.is_empty() {
            return Ok(false);
        }

        doit(&*self.stack, key.into())
    }
}

#[cfg(test)]
mod test {
    use crate::plain::Cache as PlainCache;
    use crate::sharded::Cache as ShardedCache;
    use crate::Key;
    use crate::ReadOnlyCache;
    use crate::ReadOnlyCacheBuilder;

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

    /// A stack of 0 caches should always succeed with a trivial result.
    #[test]
    fn empty() {
        let ro: ReadOnlyCache = Default::default();

        assert!(matches!(ro.get(Key::new("foo", 1, 2)), Ok(None)));
        assert!(matches!(ro.touch(Key::new("foo", 1, 2)), Ok(false)));
    }

    /// Populate a plain and a sharded cache. We should be able to access
    /// both.
    #[test]
    fn smoke_test() {
        use std::io::{Read, Write};
        use tempfile::NamedTempFile;
        use test_dir::{DirBuilder, FileType, TestDir};

        let temp = TestDir::temp()
            .create("sharded", FileType::Dir)
            .create("plain", FileType::Dir);

        {
            let cache = ShardedCache::new(temp.path("sharded"), 10, 20);

            let tmp = NamedTempFile::new_in(cache.temp_dir(None).expect("temp_dir must succeed"))
                .expect("new temp file must succeed");
            tmp.as_file()
                .write_all(b"sharded")
                .expect("write must succeed");

            cache
                .put(Key::new("a", 0, 1), tmp.path())
                .expect("put must succeed");

            let tmp2 = NamedTempFile::new_in(cache.temp_dir(None).expect("temp_dir must succeed"))
                .expect("new temp file must succeed");
            tmp2.as_file()
                .write_all(b"sharded2")
                .expect("write must succeed");

            cache
                .put(Key::new("b", 0, 1), tmp2.path())
                .expect("put must succeed");
        }

        {
            let cache = PlainCache::new(temp.path("plain"), 10);

            let tmp = NamedTempFile::new_in(cache.temp_dir().expect("temp_dir must succeed"))
                .expect("new temp file must succeed");
            tmp.as_file()
                .write_all(b"plain")
                .expect("write must succeed");

            cache.put("b", tmp.path()).expect("put must succeed");

            let tmp2 = NamedTempFile::new_in(cache.temp_dir().expect("temp_dir must succeed"))
                .expect("new temp file must succeed");
            tmp2.as_file()
                .write_all(b"plain2")
                .expect("write must succeed");

            cache.put("c", tmp2.path()).expect("put must succeed");
        }

        // sharded.a => "sharded"
        // sharded.b => "sharded2"
        // plain.b => "plain"
        // plain.c => "plain2"

        // Read from sharded, then plain.
        {
            let ro = ReadOnlyCacheBuilder::new()
                .sharded(temp.path("sharded"), 10)
                .plain(temp.path("plain"))
                .build();

            assert!(matches!(ro.get(&TestKey::new("Missing")), Ok(None)));
            assert!(matches!(ro.touch(&TestKey::new("Missing")), Ok(false)));

            // We should be able to touch `a`.
            assert!(matches!(ro.touch(&TestKey::new("a")), Ok(true)));

            // And now check that we get the correct file contents.
            {
                let mut a_file = ro
                    .get(&TestKey::new("a"))
                    .expect("must succeed")
                    .expect("must exist");
                let mut dst = Vec::new();
                a_file.read_to_end(&mut dst).expect("read must succeed");
                assert_eq!(&dst, b"sharded");
            }

            {
                let mut b_file = ro
                    .get(&TestKey::new("b"))
                    .expect("must succeed")
                    .expect("must exist");
                let mut dst = Vec::new();
                b_file.read_to_end(&mut dst).expect("read must succeed");
                assert_eq!(&dst, b"sharded2");
            }

            {
                let mut c_file = ro
                    .get(&TestKey::new("c"))
                    .expect("must succeed")
                    .expect("must exist");
                let mut dst = Vec::new();
                c_file.read_to_end(&mut dst).expect("read must succeed");
                assert_eq!(&dst, b"plain2");
            }
        }

        // Read from plain then sharded.
        {
            let ro = ReadOnlyCacheBuilder::new()
                .cache(temp.path("plain"), 1)
                .cache(temp.path("sharded"), 10)
                .build();

            {
                let mut a_file = ro
                    .get(&TestKey::new("a"))
                    .expect("must succeed")
                    .expect("must exist");
                let mut dst = Vec::new();
                a_file.read_to_end(&mut dst).expect("read must succeed");
                assert_eq!(&dst, b"sharded");
            }

            {
                let mut b_file = ro
                    .get(&TestKey::new("b"))
                    .expect("must succeed")
                    .expect("must exist");
                let mut dst = Vec::new();
                b_file.read_to_end(&mut dst).expect("read must succeed");
                assert_eq!(&dst, b"plain");
            }

            {
                let mut c_file = ro
                    .get(&TestKey::new("c"))
                    .expect("must succeed")
                    .expect("must exist");
                let mut dst = Vec::new();
                c_file.read_to_end(&mut dst).expect("read must succeed");
                assert_eq!(&dst, b"plain2");
            }
        }
    }
}
