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

use derivative::Derivative;

use crate::plain::Cache as PlainCache;
use crate::sharded::Cache as ShardedCache;
use crate::Key;

/// A `ConsistencyChecker` function compares cached values for the
/// same key and returns `Err` when the values are incompatible.
type ConsistencyChecker = Arc<
    dyn Fn(&mut File, &mut File) -> Result<()>
        + Sync
        + Send
        + std::panic::RefUnwindSafe
        + std::panic::UnwindSafe,
>;

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
/// The default builder is a fresh builder with no constituent cache
/// and no consistency check function.
#[derive(Default, Derivative)]
#[derivative(Debug)]
pub struct ReadOnlyCacheBuilder {
    stack: Vec<Box<dyn ReadSide>>,

    #[derivative(Debug = "ignore")]
    consistency_checker: Option<ConsistencyChecker>,
}

/// A [`ReadOnlyCache`] wraps an arbitrary number of
/// [`crate::plain::Cache`] and [`crate::sharded::Cache`], and attempts
/// to satisfy [`ReadOnlyCache::get`] and [`ReadOnlyCache::touch`]
/// requests by hitting each constituent cache in order.  This
/// interface hides the difference between plain and sharded cache
/// directories, and should be the first resort for read-only uses.
///
/// The default cache wraps an empty set of constituent caches and
/// performs no consistency check.
///
/// [`ReadOnlyCache`] objects are stateless and cheap to clone; don't
/// put an [`Arc`] on them.  Avoid creating multiple
/// [`ReadOnlyCache`]s for the same stack of directories: there is no
/// internal state to maintain, so multiple instances simply waste
/// memory without any benefit.
#[derive(Clone, Derivative)]
#[derivative(Debug)]
pub struct ReadOnlyCache {
    stack: Arc<[Box<dyn ReadSide>]>,

    /// When populated, the `ReadOnlyCache` keeps searching after the
    /// first cache hit, and compares subsequent hits with the first one
    /// by calling the `consistency_checker` function.  That function
    /// should return `Ok(())` if the two files are compatible (identical),
    /// and `Err` otherwise.
    #[derivative(Debug = "ignore")]
    consistency_checker: Option<ConsistencyChecker>,
}

impl ReadOnlyCacheBuilder {
    /// Returns a fresh empty builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the consistency checker function: when the function is
    /// provided, the `ReadOnlyCache` will keep searching after the
    /// first cache hit, and compare subsequent hits with the first
    /// one by calling `checker`.  The `checker` function should
    /// return `Ok(())` if the two files are compatible (identical),
    /// and `Err` otherwise.
    ///
    /// Kismet will propagate the error on mismatch.
    pub fn consistency_checker(
        &mut self,
        checker: impl Fn(&mut File, &mut File) -> Result<()>
            + Sync
            + Send
            + std::panic::RefUnwindSafe
            + std::panic::UnwindSafe
            + Sized
            + 'static,
    ) -> &mut Self {
        self.arc_consistency_checker(Some(Arc::new(checker)))
    }

    /// Removes the consistency checker function, if any.
    pub fn clear_consistency_checker(&mut self) -> &mut Self {
        self.arc_consistency_checker(None)
    }

    /// Sets the consistency checker function.  `None` clears the
    /// checker function.  See
    /// [`ReadOnlyCacheBuilder::consistency_checker`].
    #[allow(clippy::type_complexity)] // We want the public type to be transparent
    pub fn arc_consistency_checker(
        &mut self,
        checker: Option<
            Arc<
                dyn Fn(&mut File, &mut File) -> Result<()>
                    + Sync
                    + Send
                    + std::panic::RefUnwindSafe
                    + std::panic::UnwindSafe,
            >,
        >,
    ) -> &mut Self {
        self.consistency_checker = checker;
        self
    }

    /// Adds a new cache directory at `path` to the end of the cache
    /// builder's search list.
    ///
    /// Adds a plain cache directory if `num_shards <= 1`, and an
    /// actual sharded directory otherwise.
    pub fn cache(&mut self, path: impl AsRef<Path>, num_shards: usize) -> &mut Self {
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
    pub fn plain(&mut self, path: impl AsRef<Path>) -> &mut Self {
        self.stack.push(Box::new(PlainCache::new(
            path.as_ref().to_owned(),
            usize::MAX,
        )));

        self
    }

    /// Adds a new plain cache directory for each path in `paths`.
    /// The caches are appended in order to the end of the cache
    /// builder's search list.
    pub fn plain_caches<P>(&mut self, paths: impl IntoIterator<Item = P>) -> &mut Self
    where
        P: AsRef<Path>,
    {
        for path in paths {
            self.plain(path);
        }

        self
    }

    /// Adds a new sharded cache directory at `path` to the end of the
    /// cache builder's search list.
    pub fn sharded(&mut self, path: impl AsRef<Path>, num_shards: usize) -> &mut Self {
        self.stack.push(Box::new(ShardedCache::new(
            path.as_ref().to_owned(),
            num_shards,
            usize::MAX,
        )));
        self
    }

    /// Returns the contents of `self` as a fresh value; `self` is
    /// reset to the default empty builder state.  This makes it
    /// possible to declare simple configurations in a single
    /// expression, with `.take().build()`.
    pub fn take(&mut self) -> Self {
        std::mem::take(self)
    }

    /// Returns a fresh [`ReadOnlyCache`] for the builder's search list
    /// of constituent cache directories.
    pub fn build(self) -> ReadOnlyCache {
        ReadOnlyCache::new(self.stack, self.consistency_checker)
    }
}

impl Default for ReadOnlyCache {
    fn default() -> ReadOnlyCache {
        ReadOnlyCache::new(Default::default(), None)
    }
}

impl ReadOnlyCache {
    fn new(
        stack: Vec<Box<dyn ReadSide>>,
        consistency_checker: Option<ConsistencyChecker>,
    ) -> ReadOnlyCache {
        ReadOnlyCache {
            stack: stack.into_boxed_slice().into(),
            consistency_checker,
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
        fn doit(
            stack: &[Box<dyn ReadSide>],
            checker: &Option<ConsistencyChecker>,
            key: Key,
        ) -> Result<Option<File>> {
            use std::io::Seek;
            use std::io::SeekFrom;

            let mut ret = None;
            for cache in stack.iter() {
                let mut hit = match cache.get(key)? {
                    Some(hit) => hit,
                    None => continue,
                };

                match checker {
                    None => return Ok(Some(hit)),
                    Some(checker) => match ret.as_mut() {
                        None => ret = Some(hit),
                        Some(prev) => {
                            checker(prev, &mut hit)?;
                            prev.seek(SeekFrom::Start(0))?;
                        }
                    },
                }
            }

            Ok(ret)
        }

        if self.stack.is_empty() {
            return Ok(None);
        }

        doit(&*self.stack, &self.consistency_checker, key.into())
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
    use std::fs::File;
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

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

    fn byte_equality_checker(
        counter: Arc<AtomicU64>,
    ) -> impl 'static + Fn(&mut File, &mut File) -> std::io::Result<()> {
        move |x: &mut File, y: &mut File| {
            counter.fetch_add(1, Ordering::Relaxed);
            crate::byte_equality_checker(x, y)
        }
    }

    /// A stack of 0 caches should always succeed with a trivial result.
    #[test]
    fn empty() {
        let ro: ReadOnlyCache = Default::default();

        assert!(matches!(ro.get(Key::new("foo", 1, 2)), Ok(None)));
        assert!(matches!(ro.touch(Key::new("foo", 1, 2)), Ok(false)));
    }

    /// Populate two plain caches and set a consistency checker.  We
    /// should access both.
    #[test]
    fn consistency_checker_success() {
        use std::io::Read;
        use test_dir::{DirBuilder, FileType, TestDir};

        let temp = TestDir::temp()
            .create("first", FileType::Dir)
            .create("second", FileType::Dir)
            .create("first/0", FileType::ZeroFile(2))
            .create("second/0", FileType::ZeroFile(2))
            .create("first/1", FileType::RandomFile(10))
            .create("second/2", FileType::RandomFile(10));

        let counter = Arc::new(AtomicU64::new(0));

        let ro = ReadOnlyCacheBuilder::new()
            .plain(temp.path("first"))
            .plain(temp.path("second"))
            .consistency_checker(byte_equality_checker(counter.clone()))
            .take()
            .build();

        let mut hit = ro
            .get(&TestKey::new("0"))
            .expect("must succeed")
            .expect("must exist");

        assert_eq!(counter.load(Ordering::Relaxed), 1);

        let mut contents = Vec::new();
        hit.read_to_end(&mut contents).expect("read should succeed");
        assert_eq!(contents, "00".as_bytes());

        let _ = ro
            .get(&TestKey::new("1"))
            .expect("must succeed")
            .expect("must exist");
        // Only found in one subcache, there's nothing to check.
        assert_eq!(counter.load(Ordering::Relaxed), 1);

        let _ = ro
            .get(&TestKey::new("2"))
            .expect("must succeed")
            .expect("must exist");
        // Only found in one subcache, there's nothing to check.
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }

    /// Populate two plain caches and set a consistency checker.  We
    /// should error on mismatch.
    #[test]
    fn consistency_checker_failure() {
        use test_dir::{DirBuilder, FileType, TestDir};

        let temp = TestDir::temp()
            .create("first", FileType::Dir)
            .create("second", FileType::Dir)
            .create("first/0", FileType::ZeroFile(2))
            .create("second/0", FileType::ZeroFile(3));

        let counter = Arc::new(AtomicU64::new(0));
        let ro = ReadOnlyCacheBuilder::new()
            .plain(temp.path("first"))
            .plain(temp.path("second"))
            .consistency_checker(byte_equality_checker(counter))
            .take()
            .build();

        // This call should error.
        assert!(ro.get(&TestKey::new("0")).is_err());
    }

    /// Populate two plain caches and unset the consistency checker.  We
    /// should not error.
    #[test]
    fn consistency_checker_silent_failure() {
        use test_dir::{DirBuilder, FileType, TestDir};

        let temp = TestDir::temp()
            .create("first", FileType::Dir)
            .create("second", FileType::Dir)
            .create("first/0", FileType::ZeroFile(2))
            .create("second/0", FileType::ZeroFile(3));

        let counter = Arc::new(AtomicU64::new(0));

        let ro = ReadOnlyCacheBuilder::new()
            .plain(temp.path("first"))
            .plain(temp.path("second"))
            .consistency_checker(byte_equality_checker(counter.clone()))
            .clear_consistency_checker()
            .take()
            .build();

        // This call should not error.
        let _ = ro
            .get(&TestKey::new("0"))
            .expect("must succeed")
            .expect("must exist");

        // There should be no call to the checker function.
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    /// Populate two plain caches.  We should read from both.
    #[test]
    fn two_plain_caches() {
        use test_dir::{DirBuilder, FileType, TestDir};

        let temp = TestDir::temp()
            .create("first", FileType::Dir)
            .create("second", FileType::Dir)
            .create("first/0", FileType::ZeroFile(2))
            .create("second/1", FileType::ZeroFile(3));

        let ro = ReadOnlyCacheBuilder::new()
            .plain_caches(["first", "second"].iter().map(|p| temp.path(p)))
            .take()
            .build();

        // We should find 0 and 1.
        let _ = ro
            .get(&TestKey::new("0"))
            .expect("must succeed")
            .expect("must exist");

        let _ = ro
            .get(&TestKey::new("1"))
            .expect("must succeed")
            .expect("must exist");

        // But not 2.
        assert!(ro.get(&TestKey::new("2")).expect("must succeed").is_none());
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
                .take()
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
                .take()
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
