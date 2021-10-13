//! We expect most callers to interact with Kismet via the [`Cache`]
//! struct defined here.  A [`Cache`] hides the difference in
//! behaviour between [`crate::plain::Cache`] and
//! [`crate::sharded::Cache`] via late binding, and lets callers
//! transparently handle misses by looking in a series of secondary
//! cache directories.
use std::borrow::Cow;
use std::fs::File;
use std::io::Error;
use std::io::ErrorKind;
use std::io::Result;
use std::path::Path;
use std::sync::Arc;

use derivative::Derivative;
use tempfile::NamedTempFile;

use crate::plain::Cache as PlainCache;
use crate::sharded::Cache as ShardedCache;
use crate::Key;
use crate::ReadOnlyCache;
use crate::ReadOnlyCacheBuilder;

/// A `ConsistencyChecker` function compares cached values for the
/// same key and returns `Err` when the values are incompatible.
type ConsistencyChecker = Arc<
    dyn Fn(&mut File, &mut File) -> Result<()>
        + Sync
        + Send
        + std::panic::RefUnwindSafe
        + std::panic::UnwindSafe,
>;

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

/// Construct a [`Cache`] with this builder.  The resulting cache will
/// always first access its write-side cache (if defined), and, on
/// misses, will attempt to service [`Cache::get`] and
/// [`Cache::touch`] calls by iterating over the read-only caches.
#[derive(Derivative)]
#[derivative(Debug)]
pub struct CacheBuilder {
    write_side: Option<Arc<dyn FullCache>>,
    auto_sync: bool,

    #[derivative(Debug = "ignore")]
    consistency_checker: Option<ConsistencyChecker>,

    read_side: ReadOnlyCacheBuilder,
}

impl Default for CacheBuilder {
    fn default() -> CacheBuilder {
        CacheBuilder {
            write_side: None,
            auto_sync: true,
            consistency_checker: None,
            read_side: Default::default(),
        }
    }
}

/// A [`Cache`] wraps either up to one plain or sharded read-write
/// cache in a convenient interface, and may optionally fulfill read
/// operations by deferring to a list of read-only cache when the
/// read-write cache misses.
///
/// The default cache has no write-side and an empty stack of backup
/// read-only caches.
///
/// [`Cache`] objects are cheap to clone and lock-free; don't put an
/// [`Arc`] on them.  Avoid opening multiple caches for the same set
/// of directories: using the same [`Cache`] object improves the
/// accuracy of the write cache's lock-free in-memory statistics, when
/// it's a sharded cache.
#[derive(Clone, Derivative)]
#[derivative(Debug)]
pub struct Cache {
    // The write-side cache services writes and is the cache of first
    // resort for `get` and `touch`.
    write_side: Option<Arc<dyn FullCache>>,
    // Whether to automatically sync file contents before publishing
    // them to the write-side cache.
    auto_sync: bool,

    // If provided, `Kismet` will compare results to make sure all
    // cache levels that have a value for a given key agree ( the
    // checker function returns `Ok(())`).
    #[derivative(Debug = "ignore")]
    consistency_checker: Option<ConsistencyChecker>,

    // The read-side cache (a list of read-only caches) services `get`
    // and `touch` calls when we fail to find something in the
    // write-side cache.
    read_side: ReadOnlyCache,
}

impl Default for Cache {
    fn default() -> Cache {
        Cache {
            write_side: None,
            auto_sync: true,
            consistency_checker: None,
            read_side: Default::default(),
        }
    }
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

/// What to do with a cache hit in a [`Cache::get_or_update`] call?
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
    /// checker function.  See [`CacheBuilder::consistency_checker`].
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
        self.consistency_checker = checker.clone();
        self.read_side.arc_consistency_checker(checker);
        self
    }

    /// Sets the read-write cache directory to `path`.
    ///
    /// The read-write cache will be a plain cache directory if
    /// `num_shards <= 1`, and a sharded directory otherwise.
    pub fn writer(
        &mut self,
        path: impl AsRef<Path>,
        num_shards: usize,
        total_capacity: usize,
    ) -> &mut Self {
        if num_shards <= 1 {
            self.plain_writer(path, total_capacity)
        } else {
            self.sharded_writer(path, num_shards, total_capacity)
        }
    }

    /// Sets the read-write cache directory to a plain directory at
    /// `path`, with a target file count of up to `capacity`.
    pub fn plain_writer(&mut self, path: impl AsRef<Path>, capacity: usize) -> &mut Self {
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
        &mut self,
        path: impl AsRef<Path>,
        num_shards: usize,
        total_capacity: usize,
    ) -> &mut Self {
        self.write_side.insert(Arc::new(ShardedCache::new(
            path.as_ref().to_owned(),
            num_shards,
            total_capacity,
        )));
        self
    }

    /// Sets whether files published read-write cache will be
    /// automatically flushed to disk with [`File::sync_all`]
    /// before sending them to the cache directory.
    ///
    /// Defaults to true, for safety.  Even when `auto_sync` is
    /// enabled, Kismet does not `fsync` cache directories; after a
    /// kernel or hardware crash, caches may partially revert to an
    /// older state, but should not contain incomplete files.
    ///
    /// An application may want to disable `auto_sync` because it
    /// already synchronises files, or because the cache directories
    /// do not survive crashes: they might be erased after each boot,
    /// e.g., via
    /// [tmpfiles.d](https://www.freedesktop.org/software/systemd/man/tmpfiles.d.html),
    /// or tagged with a [boot id](https://man7.org/linux/man-pages/man3/sd_id128_get_machine.3.html).
    pub fn auto_sync(&mut self, sync: bool) -> &mut Self {
        self.auto_sync = sync;
        self
    }

    /// Adds a new read-only cache directory at `path` to the end of the
    /// cache builder's search list.
    ///
    /// Adds a plain cache directory if `num_shards <= 1`, and a sharded
    /// directory otherwise.
    pub fn reader(&mut self, path: impl AsRef<Path>, num_shards: usize) -> &mut Self {
        self.read_side.cache(path, num_shards);
        self
    }

    /// Adds a new plain (unsharded) read-only cache directory at
    /// `path` to the end of the cache builder's search list.
    pub fn plain_reader(&mut self, path: impl AsRef<Path>) -> &mut Self {
        self.read_side.plain(path);
        self
    }

    /// Adds a new plain cache read-only directory for each path in
    /// `paths`.  The caches are appended in order to the end of the
    /// cache builder's search list.
    pub fn plain_readers(
        &mut self,
        paths: impl IntoIterator<Item = impl AsRef<Path>>,
    ) -> &mut Self {
        self.read_side.plain_caches(paths);
        self
    }

    /// Adds a new sharded read-only cache directory at `path` to the
    /// end of the cache builder's search list.
    pub fn sharded_reader(&mut self, path: impl AsRef<Path>, num_shards: usize) -> &mut Self {
        self.read_side.sharded(path, num_shards);
        self
    }

    /// Returns the contents of `self` as a fresh value; `self` is
    /// reset to the default empty builder state.  This makes it
    /// possible to declare simple configurations in a single
    /// expression, with `.take().build()`.
    pub fn take(&mut self) -> Self {
        std::mem::take(self)
    }

    /// Returns a fresh [`Cache`] for the builder's write cache and
    /// additional search list of read-only cache directories.
    pub fn build(self) -> Cache {
        Cache {
            write_side: self.write_side,
            auto_sync: self.auto_sync,
            consistency_checker: self.consistency_checker,
            read_side: self.read_side.build(),
        }
    }
}

impl Cache {
    /// Calls [`File::sync_all`] on `file` if `Cache::auto_sync`
    /// is true.
    #[inline]
    fn maybe_sync(&self, file: &File) -> Result<()> {
        if self.auto_sync {
            file.sync_all()
        } else {
            Ok(())
        }
    }

    /// Opens `path` and calls [`File::sync_all`] on the resulting
    /// file, if `Cache::auto_sync` is true.
    ///
    /// Panics when [`File::sync_all`] fails. See
    /// https://wiki.postgresql.org/wiki/Fsync_Errors or
    /// Rebello et al's "Can Applications Recover from fsync Failures?"
    /// (https://www.usenix.org/system/files/atc20-rebello.pdf)
    /// for an idea of the challenges associated with handling
    /// fsync failures on persistent files.
    fn maybe_sync_path(&self, path: &Path) -> Result<()> {
        if self.auto_sync {
            // It's really not clear what happens to a file's content
            // if we open it just before fsync, and fsync fails.  It
            // should be safe to just unlink the file
            std::fs::File::open(&path)?
                .sync_all()
                .expect("auto_sync failed, and failure semantics are unclear for fsync");
        }

        Ok(())
    }

    /// Attempts to open a read-only file for `key`.  The `Cache` will
    /// query each its write cache (if any), followed by the list of
    /// additional read-only cache, in definition order, and return a
    /// read-only file for the first hit.
    ///
    /// Fails with [`ErrorKind::InvalidInput`] if `key.name` is invalid
    /// (empty, or starts with a dot or a forward or back slash).
    ///
    /// Returns [`None`] if no file for `key` can be found in any of the
    /// constituent caches, and bubbles up the first I/O error
    /// encountered, if any.
    ///
    /// In the worst case, each call to `get` attempts to open two
    /// files for the [`Cache`]'s read-write directory and for each
    /// read-only backup directory.
    pub fn get<'a>(&self, key: impl Into<Key<'a>>) -> Result<Option<File>> {
        fn doit(
            write_side: Option<&dyn FullCache>,
            checker: Option<&ConsistencyChecker>,
            read_side: &ReadOnlyCache,
            key: Key,
        ) -> Result<Option<File>> {
            use std::io::Seek;
            use std::io::SeekFrom;

            if let Some(write) = write_side {
                if let Some(mut ret) = write.get(key)? {
                    if let Some(checker) = checker {
                        if let Some(mut read_hit) = read_side.get(key)? {
                            checker(&mut ret, &mut read_hit)?;
                            ret.seek(SeekFrom::Start(0))?;
                        }
                    }

                    return Ok(Some(ret));
                }
            }

            read_side.get(key)
        }

        doit(
            self.write_side.as_ref().map(AsRef::as_ref),
            self.consistency_checker.as_ref(),
            &self.read_side,
            key.into(),
        )
    }

    /// Attempts to find a cache entry for `key`.  If there is none,
    /// populates the cache with a file filled by `populate`.  Returns
    /// a file in all cases (unless the call fails with an error).
    ///
    /// Always invokes `populate` for a consistency check when a
    /// consistency check function is provided.  The `populate`
    /// function can return `ErrorKind::NotFound` to skip the
    /// comparison without failing the whole call.
    ///
    /// Fails with [`ErrorKind::InvalidInput`] if `key.name` is
    /// invalid (empty, or starts with a dot or a forward or back slash).
    ///
    /// See [`Cache::get_or_update`] for more control over the operation.
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
    /// Always invokes `populate` for a consistency check when a
    /// consistency check function is provided.  The `populate`
    /// function can return `ErrorKind::NotFound` to skip the
    /// comparison without failing the whole call.
    ///
    /// Fails with [`ErrorKind::InvalidInput`] if `key.name` is
    /// invalid (empty, or starts with a dot or a forward or back slash).
    ///
    /// When we need to populate a new file, `populate` is called with
    /// a mutable reference to the destination file, and the old
    /// cached file (in whatever state `judge` left it), if available.
    ///
    /// See [`Cache::ensure`] for a simpler interface.
    ///
    /// In the worst case, each call to `get_or_update` attempts to
    /// open two files for the [`Cache`]'s read-write directory and
    /// for each read-only backup directory, and fails to find
    /// anything.  `get_or_update` then publishes a new cached file
    /// (in a constant number of file operations), but not before
    /// triggering a second chance maintenance (time linearithmic in
    /// the number of files in the directory chosen for maintenance,
    /// but amortised to logarithmic).
    pub fn get_or_update<'a>(
        &self,
        key: impl Into<Key<'a>>,
        judge: impl FnOnce(CacheHit) -> CacheHitAction,
        populate: impl FnOnce(&mut File, Option<File>) -> Result<()>,
    ) -> Result<File> {
        use std::io::Seek;
        use std::io::SeekFrom;

        // Promotes `file` to `cache`.
        fn promote(cache: &dyn FullCache, sync: bool, key: Key, mut file: File) -> Result<File> {
            let mut tmp = NamedTempFile::new_in(cache.temp_dir(key)?)?;
            std::io::copy(&mut file, tmp.as_file_mut())?;

            // Force the destination file's contents to disk before
            // adding it to the read-write cache, if we're supposed to
            // sync files automatically.
            if sync {
                tmp.as_file().sync_all()?;
            }

            cache.put(key, tmp.path())?;

            // We got a read-only file.  Rewind it before returning.
            file.seek(SeekFrom::Start(0))?;
            Ok(file)
        }

        let cache_or = self.write_side.as_ref().map(Arc::as_ref);
        let key: Key = key.into();

        let get_tempfile = || {
            if let Some(cache) = cache_or {
                tempfile::tempfile_in(cache.temp_dir(key)?)
            } else {
                tempfile::tempfile()
            }
        };

        let mut old = None; // Overwritten with `Some(file)` when replacing `file`.
        if let Some(mut file) = cache_or
            .and_then(|cache| cache.get(key).transpose())
            .transpose()?
        {
            if let Some(checker) = self.consistency_checker.as_ref() {
                if let Some(mut read) = self.read_side.get(key)? {
                    checker(&mut file, &mut read)?;
                    file.seek(SeekFrom::Start(0))?;
                }
            }

            match judge(CacheHit::Primary(&mut file)) {
                // Promote is a no-op if the file is already in the write cache.
                CacheHitAction::Accept | CacheHitAction::Promote => {
                    file.seek(SeekFrom::Start(0))?;

                    if let Some(checker) = self.consistency_checker.as_ref() {
                        let mut tmp = get_tempfile()?;
                        match populate(&mut tmp, None) {
                            Err(e) if e.kind() == ErrorKind::NotFound => {
                                return Ok(file);
                            }
                            ret => ret?,
                        };
                        tmp.seek(SeekFrom::Start(0))?;
                        checker(&mut file, &mut tmp)?;
                        file.seek(SeekFrom::Start(0))?;
                    }

                    return Ok(file);
                }
                CacheHitAction::Replace => old = Some(file),
            }
        } else if let Some(mut file) = self.read_side.get(key)? {
            match judge(CacheHit::Secondary(&mut file)) {
                j @ CacheHitAction::Accept | j @ CacheHitAction::Promote => {
                    file.seek(SeekFrom::Start(0))?;

                    if let Some(checker) = self.consistency_checker.as_ref() {
                        let mut tmp = get_tempfile()?;

                        match populate(&mut tmp, None) {
                            Err(e) if e.kind() == ErrorKind::NotFound => {
                                return Ok(file);
                            }
                            ret => ret?,
                        };

                        tmp.seek(SeekFrom::Start(0))?;
                        checker(&mut file, &mut tmp)?;
                        file.seek(SeekFrom::Start(0))?;
                    }

                    return if matches!(j, CacheHitAction::Accept) {
                        Ok(file)
                    } else if let Some(cache) = cache_or {
                        promote(cache, self.auto_sync, key, file)
                    } else {
                        Ok(file)
                    };
                }
                CacheHitAction::Replace => old = Some(file),
            }
        }

        let cache = match cache_or {
            Some(cache) => cache,
            None => {
                // If there's no write-side cache, satisfy the cache miss
                // without saving the result anywhere.
                let mut tmp = tempfile::tempfile()?;
                populate(&mut tmp, old)?;

                tmp.seek(SeekFrom::Start(0))?;
                return Ok(tmp);
            }
        };

        let replace = old.is_some();
        // We either have to replace or ensure there is a cache entry.
        // Either way, start by populating a temporary file.
        let mut tmp = NamedTempFile::new_in(cache.temp_dir(key)?)?;
        populate(tmp.as_file_mut(), old)?;
        self.maybe_sync(tmp.as_file())?;

        // Grab a read-only return value before publishing the file.
        let path = tmp.path();
        let mut ret = File::open(path)?;
        if replace {
            cache.set(key, path)?;
        } else {
            cache.put(key, path)?;
            // Return the now-cached file, if we can get it.
            if let Ok(Some(file)) = cache.get(key) {
                ret = file;
            }
        }

        Ok(ret)
    }

    fn set_impl(&self, key: Key, value: &Path) -> Result<()> {
        match self.write_side.as_ref() {
            Some(write) => write.set(key, value),
            None => Err(Error::new(
                ErrorKind::Unsupported,
                "no kismet write cache defined",
            )),
        }
    }

    /// Inserts or overwrites the file at `value` as `key` in the
    /// write cache directory.  This will always fail with
    /// [`ErrorKind::Unsupported`] if no write cache was defined.
    /// The path at `value` must be in the same filesystem as the
    /// write cache directory: we rely on atomic file renames.
    ///
    /// Fails with [`ErrorKind::InvalidInput`] if `key.name` is invalid
    /// (empty, or starts with a dot or a forward or back slash).
    ///
    /// Always consumes the file at `value` on success; may consume it
    /// on error.
    ///
    /// When `auto_sync` is enabled (the default), the file at `value`
    /// will always be [`File::sync_all`]ed before publishing to the
    /// cache.  Kismet will **panic** when the [`File::sync_all`] call
    /// itself fails: retrying the same call to [`Cache::set`] could
    /// erroneously succeed, since some filesystems clear internal I/O
    /// failure flag after the first `fsync`.
    ///
    /// Executes in a bounded number of file operations, except for
    /// the lock-free maintenance, which needs time linearithmic in
    /// the number of files in the directory chosen for maintenance,
    /// amortised to logarithmic, and constant number of file operations.
    pub fn set<'a>(&self, key: impl Into<Key<'a>>, value: impl AsRef<Path>) -> Result<()> {
        fn doit(this: &Cache, key: Key, value: &Path) -> Result<()> {
            this.maybe_sync_path(value)?;
            this.set_impl(key, value)
        }

        doit(self, key.into(), value.as_ref())
    }

    /// Invokes [`Cache::set`] on a [`tempfile::NamedTempFile`].
    ///
    /// See [`Cache::set`] for more details.  The only difference is
    /// that `set_temp_file` does not panic when `auto_sync` is enabled
    /// and we fail to [`File::sync_all`] the [`NamedTempFile`] value.
    pub fn set_temp_file<'a>(&self, key: impl Into<Key<'a>>, value: NamedTempFile) -> Result<()> {
        fn doit(this: &Cache, key: Key, value: NamedTempFile) -> Result<()> {
            this.maybe_sync(value.as_file())?;
            this.set_impl(key, value.path())
        }

        doit(self, key.into(), value)
    }

    fn put_impl(&self, key: Key, value: &Path) -> Result<()> {
        match self.write_side.as_ref() {
            Some(write) => write.put(key, value),
            None => Err(Error::new(
                ErrorKind::Unsupported,
                "no kismet write cache defined",
            )),
        }
    }

    /// Inserts the file at `value` as `key` in the cache directory if
    /// there is no such cached entry already, or touches the cached
    /// file if it already exists.  This will always fail with
    /// [`ErrorKind::Unsupported`] if no write cache was defined.
    /// The path at `value` must be in the same filesystem as the
    /// write cache directory: we rely on atomic file hard linkage.
    ///
    /// Fails with [`ErrorKind::InvalidInput`] if `key.name` is invalid
    /// (empty, or starts with a dot or a forward or back slash).
    ///
    /// Always consumes the file at `value` on success; may consume it
    /// on error.
    ///
    /// When `auto_sync` is enabled (the default), the file at `value`
    /// will always be [`File::sync_all`]ed before publishing to the
    /// cache.  Kismet will **panic** when the [`File::sync_all`] call
    /// itself fails: retrying the same call to [`Cache::put`] could
    /// erroneously succeed, since some filesystems clear internal I/O
    /// failure flag after the first `fsync`.
    ///
    /// Executes in a bounded number of file operations, except for
    /// the lock-free maintenance, which needs time linearithmic in
    /// the number of files in the directory chosen for maintenance,
    /// amortised to logarithmic, and constant number of file operations.
    pub fn put<'a>(&self, key: impl Into<Key<'a>>, value: impl AsRef<Path>) -> Result<()> {
        fn doit(this: &Cache, key: Key, value: &Path) -> Result<()> {
            this.maybe_sync_path(value)?;
            this.put_impl(key, value)
        }

        doit(self, key.into(), value.as_ref())
    }

    /// Invokes [`Cache::put`] on a [`tempfile::NamedTempFile`].
    ///
    /// See [`Cache::put`] for more details.  The only difference is
    /// that `put_temp_file` does not panic when `auto_sync` is enabled
    /// and we fail to [`File::sync_all`] the [`NamedTempFile`] value.
    pub fn put_temp_file<'a>(&self, key: impl Into<Key<'a>>, value: NamedTempFile) -> Result<()> {
        fn doit(this: &Cache, key: Key, value: NamedTempFile) -> Result<()> {
            this.maybe_sync(value.as_file())?;
            this.put_impl(key, value.path())
        }

        doit(self, key.into(), value)
    }

    /// Marks a cache entry for `key` as accessed (read).  The [`Cache`]
    /// will touch the same file that would be returned by `get`.
    ///
    /// Fails with [`ErrorKind::InvalidInput`] if `key.name` is invalid
    /// (empty, or starts with a dot or a forward or back slash).
    ///
    /// Returns whether a file for `key` could be found, and bubbles
    /// up the first I/O error encountered, if any.
    ///
    /// In the worst case, each call to `touch` attempts to update the
    /// access time on two files for each cache directory in the
    /// `ReadOnlyCache` stack.
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
    use std::fs::File;
    use std::io::ErrorKind;
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    use crate::plain::Cache as PlainCache;
    use crate::sharded::Cache as ShardedCache;
    use crate::Cache;
    use crate::CacheBuilder;
    use crate::CacheHit;
    use crate::CacheHitAction;
    use crate::Key;

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
        use std::io::Read;

        move |x: &mut File, y: &mut File| {
            let mut x_contents = Vec::new();
            let mut y_contents = Vec::new();

            counter.fetch_add(1, Ordering::Relaxed);
            x.read_to_end(&mut x_contents)?;
            y.read_to_end(&mut y_contents)?;

            if x_contents == y_contents {
                Ok(())
            } else {
                Err(std::io::Error::new(std::io::ErrorKind::Other, "mismatch"))
            }
        }
    }

    // No cache defined -> read calls should successfully do nothing,
    // write calls should fail.
    #[test]
    fn empty() {
        use test_dir::{DirBuilder, FileType, TestDir};

        let temp = TestDir::temp().create("foo", FileType::RandomFile(10));
        let cache: Cache = Default::default();

        assert!(matches!(cache.get(&TestKey::new("foo")), Ok(None)));
        // Ensure also succeeds: the temporary value is recreated for
        // each miss.
        assert!(matches!(
            cache.ensure(&TestKey::new("foo"), |_| Ok(())),
            Ok(_)
        ));

        assert!(matches!(cache.set(&TestKey::new("foo"), &temp.path("foo")),
                         Err(e) if e.kind() == ErrorKind::Unsupported));
        assert!(matches!(cache.put(&TestKey::new("foo"), &temp.path("foo")),
                         Err(e) if e.kind() == ErrorKind::Unsupported));
        assert!(matches!(cache.touch(&TestKey::new("foo")), Ok(false)));
    }

    // Disable autosync; we should get an `Unsupported` error even if the
    // input file does not exist.
    #[test]
    fn empty_no_auto_sync() {
        let cache = CacheBuilder::new().auto_sync(false).take().build();

        assert!(matches!(cache.get(&TestKey::new("foo")), Ok(None)));
        assert!(matches!(
            cache.ensure(&TestKey::new("foo"), |_| Ok(())),
            Ok(_)
        ));

        assert!(
            matches!(cache.set(&TestKey::new("foo"), "/no-such-tmp/foo"),
                     Err(e) if e.kind() == ErrorKind::Unsupported)
        );
        assert!(
            matches!(cache.put(&TestKey::new("foo"), "/no-such-tmp/foo"),
                     Err(e) if e.kind() == ErrorKind::Unsupported)
        );
        assert!(matches!(cache.touch(&TestKey::new("foo")), Ok(false)));
    }

    /// Populate two plain caches and set a consistency checker.  We
    /// should access both.
    #[test]
    fn consistency_checker_success() {
        use std::io::Error;
        use std::io::ErrorKind;
        use std::io::Read;
        use std::io::Write;
        use test_dir::{DirBuilder, FileType, TestDir};

        let temp = TestDir::temp()
            .create("first", FileType::Dir)
            .create("second", FileType::Dir)
            .create("first/0", FileType::ZeroFile(2))
            .create("second/0", FileType::ZeroFile(2))
            .create("first/1", FileType::ZeroFile(1))
            .create("second/2", FileType::ZeroFile(3))
            .create("second/3", FileType::ZeroFile(3))
            .create("second/4", FileType::ZeroFile(4));

        let counter = Arc::new(AtomicU64::new(0));

        let cache = CacheBuilder::new()
            .plain_writer(temp.path("first"), 100)
            .plain_reader(temp.path("second"))
            .consistency_checker(byte_equality_checker(counter.clone()))
            .take()
            .build();

        // Find a hit in both caches. The checker should be invoked.
        {
            let mut hit = cache
                .get(&TestKey::new("0"))
                .expect("must succeed")
                .expect("must exist");

            assert_eq!(counter.load(Ordering::Relaxed), 1);

            let mut contents = Vec::new();
            hit.read_to_end(&mut contents).expect("read should succeed");
            assert_eq!(contents, "00".as_bytes());
        }

        // Do the same via `ensure`.
        {
            counter.store(0, Ordering::Relaxed);
            let mut populated = cache
                .ensure(&TestKey::new("0"), |dst| {
                    dst.write_all("00".as_bytes())?;
                    Ok(())
                })
                .expect("ensure must succeed");

            assert_eq!(counter.load(Ordering::Relaxed), 2);

            let mut contents = Vec::new();
            populated
                .read_to_end(&mut contents)
                .expect("read should succeed");
            assert_eq!(contents, "00".as_bytes());
        }

        // Now return `NotFound` from the `populate` callback,
        // we should still succeed.
        {
            counter.store(0, Ordering::Relaxed);
            let mut populated = cache
                .ensure(&TestKey::new("0"), |_| {
                    Err(Error::new(ErrorKind::NotFound, "not found"))
                })
                .expect("ensure must succeed");

            assert_eq!(counter.load(Ordering::Relaxed), 1);

            let mut contents = Vec::new();
            populated
                .read_to_end(&mut contents)
                .expect("read should succeed");
            assert_eq!(contents, "00".as_bytes());
        }

        counter.store(0, Ordering::Relaxed);
        let _ = cache
            .get(&TestKey::new("1"))
            .expect("must succeed")
            .expect("must exist");
        // Only found in the writer, there's nothing to check.
        assert_eq!(counter.load(Ordering::Relaxed), 0);

        // Do the same via `ensure`.
        {
            let mut populated = cache
                .ensure(&TestKey::new("1"), |dst| {
                    dst.write_all("0".as_bytes())?;
                    Ok(())
                })
                .expect("ensure must succeed");

            assert_eq!(counter.load(Ordering::Relaxed), 1);

            let mut contents = Vec::new();
            populated
                .read_to_end(&mut contents)
                .expect("read should succeed");
            assert_eq!(contents, "0".as_bytes());
        }

        counter.store(0, Ordering::Relaxed);
        let _ = cache
            .get(&TestKey::new("2"))
            .expect("must succeed")
            .expect("must exist");
        // Only found in the read cache, there's nothing to check.
        assert_eq!(counter.load(Ordering::Relaxed), 0);

        // Do the same via `ensure`.
        {
            counter.store(0, Ordering::Relaxed);
            let mut populated = cache
                .ensure(&TestKey::new("2"), |dst| {
                    dst.write_all("000".as_bytes())?;
                    Ok(())
                })
                .expect("ensure must succeed");

            assert_eq!(counter.load(Ordering::Relaxed), 1);

            let mut contents = Vec::new();
            populated
                .read_to_end(&mut contents)
                .expect("read should succeed");
            assert_eq!(contents, "000".as_bytes());
        }

        {
            counter.store(0, Ordering::Relaxed);
            let mut populated = cache
                .get_or_update(
                    &TestKey::new("3"),
                    |_| CacheHitAction::Accept,
                    |dst, _| {
                        dst.write_all("000".as_bytes())?;
                        Ok(())
                    },
                )
                .expect("get_or_update must succeed");

            assert_eq!(counter.load(Ordering::Relaxed), 1);

            let mut contents = Vec::new();
            populated
                .read_to_end(&mut contents)
                .expect("read should succeed");
            assert_eq!(contents, "000".as_bytes());
        }

        // Again, but now the `populate` callback returns `NotFound`.
        {
            counter.store(0, Ordering::Relaxed);
            let mut populated = cache
                .get_or_update(
                    &TestKey::new("4"),
                    |_| CacheHitAction::Accept,
                    |_, _| Err(Error::new(ErrorKind::NotFound, "not found")),
                )
                .expect("get_or_update must succeed");

            assert_eq!(counter.load(Ordering::Relaxed), 0);

            let mut contents = Vec::new();
            populated
                .read_to_end(&mut contents)
                .expect("read should succeed");
            assert_eq!(contents, "0000".as_bytes());
        }

        // Make sure we succeed on plain misses.
        {
            counter.store(0, Ordering::Relaxed);
            let mut populated = cache
                .get_or_update(
                    &TestKey::new("no-such-key"),
                    |_| CacheHitAction::Accept,
                    |dst, _| {
                        dst.write_all("fresh data".as_bytes())?;
                        Ok(())
                    },
                )
                .expect("get_or_update must succeed");

            assert_eq!(counter.load(Ordering::Relaxed), 0);

            let mut contents = Vec::new();
            populated
                .read_to_end(&mut contents)
                .expect("read should succeed");
            assert_eq!(contents, "fresh data".as_bytes());
        }
    }

    /// Populate two plain caches and set a consistency checker.  We
    /// should error on mismatch.
    #[test]
    fn consistency_checker_failure() {
        use std::io::Write;
        use test_dir::{DirBuilder, FileType, TestDir};

        let temp = TestDir::temp()
            .create("first", FileType::Dir)
            .create("second", FileType::Dir)
            .create("first/0", FileType::ZeroFile(2))
            .create("second/0", FileType::ZeroFile(3))
            .create("first/1", FileType::ZeroFile(1))
            .create("second/2", FileType::ZeroFile(4));

        let counter = Arc::new(AtomicU64::new(0));
        let cache = CacheBuilder::new()
            .plain_writer(temp.path("first"), 100)
            .plain_reader(temp.path("second"))
            .consistency_checker(byte_equality_checker(counter))
            .take()
            .build();

        // This call should error.
        assert!(cache.get(&TestKey::new("0")).is_err());

        // The call should also error through `ensure`.
        assert!(cache
            .ensure(&TestKey::new("0"), |_| {
                unreachable!("should detect read-cache mismatch first");
            })
            .is_err());

        // Do the same for the files that are only in one of the two
        // caches.
        assert!(cache
            .ensure(&TestKey::new("1"), |dst| {
                dst.write_all("0000".as_bytes())?;
                Ok(())
            })
            .is_err());

        assert!(cache
            .ensure(&TestKey::new("2"), |dst| {
                dst.write_all("0".as_bytes())?;
                Ok(())
            })
            .is_err());

        // Same with `get_or_update`.
        assert!(cache
            .get_or_update(
                &TestKey::new("2"),
                |_| CacheHitAction::Accept,
                |dst, _| {
                    dst.write_all("0".as_bytes())?;
                    Ok(())
                }
            )
            .is_err());
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
            .create("second/0", FileType::ZeroFile(3))
            .create("first/1", FileType::ZeroFile(1))
            .create("second/2", FileType::ZeroFile(4));

        let counter = Arc::new(AtomicU64::new(0));

        let cache = CacheBuilder::new()
            .plain_writer(temp.path("first"), 100)
            .plain_reader(temp.path("second"))
            .consistency_checker(byte_equality_checker(counter.clone()))
            .clear_consistency_checker()
            .take()
            .build();

        // This call should not error.
        let _ = cache
            .get(&TestKey::new("0"))
            .expect("must succeed")
            .expect("must exist");

        // And same for `ensure` calls.
        let _ = cache
            .ensure(&TestKey::new("0"), |_| {
                unreachable!("should not be called");
            })
            .expect("must succeed");

        let _ = cache
            .ensure(&TestKey::new("1"), |_| {
                unreachable!("should not be called");
            })
            .expect("must succeed");

        let _ = cache
            .ensure(&TestKey::new("2"), |_| {
                unreachable!("should not be called");
            })
            .expect("must succeed");

        let _ = cache
            .get_or_update(
                &TestKey::new("2"),
                |_| CacheHitAction::Accept,
                |_, _| {
                    unreachable!("should not be called");
                },
            )
            .expect("must succeed");
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

        let ro = CacheBuilder::new()
            .plain_readers(["first", "second"].iter().map(|p| temp.path(p)))
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

    // Fail to find a file, ensure it, then see that we can get it.
    #[test]
    fn test_ensure() {
        use std::io::{Read, Write};
        use test_dir::{DirBuilder, TestDir};

        let temp = TestDir::temp();
        // Get some coverage for no-auto_sync config.
        let cache = CacheBuilder::new()
            .writer(temp.path("."), 1, 10)
            .auto_sync(false)
            .take()
            .build();
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
            .take()
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
                .take()
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
            .take()
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
                .take()
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
            .take()
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
                .take()
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

    // Use a one-level cache, without any write-side cache.  We should
    // still be able to `ensure` (or `get_or_update`), by calling the
    // `populate` function for all misses.
    #[test]
    fn test_ensure_no_write_side() {
        use std::io::{Read, Write};
        use tempfile::NamedTempFile;
        use test_dir::{DirBuilder, FileType, TestDir};

        let temp = TestDir::temp().create("extra_plain", FileType::Dir);

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
            .plain_reader(temp.path("extra_plain"))
            .take()
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

        // Let's try again with the same file file.
        {
            let mut fetched = cache
                .get_or_update(
                    &key2,
                    |_| unreachable!("should not be called"),
                    |file, old| {
                        assert!(old.is_none());
                        file.write_all(b"updated2")
                    },
                )
                .expect("get_or_update must succeed");

            let mut dst = Vec::new();
            fetched.read_to_end(&mut dst).expect("read must succeed");
            assert_eq!(&dst, b"updated2");
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
            .take()
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
            // Exercise put_temp_file as well.
            cache
                .put_temp_file(&TestKey::new("b"), tmp)
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
            .take()
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
            // Exercise set_temp_file.
            cache
                .set_temp_file(&TestKey::new("b"), tmp)
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
