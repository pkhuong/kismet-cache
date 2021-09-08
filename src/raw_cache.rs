//! The raw cache module manages directories of read-only files
//! subject to a (batched) Second Chance eviction policy.  Calling
//! [`prune`] deletes files to make sure a cache directory does not
//! exceed its capacity, in file count.  The deletions will obey a
//! Second Chance policy as long as insertions and updates go through
//! [`insert_or_update`] or [`insert_or_touch`], in order to update the
//! cached files' modification times correctly.  Opening the cached
//! file will automatically update its metadata to take that access
//! into account, but a path can also be [`touch`]ed explicitly.
//!
//! This module implements mechanisms, but does not hardcode any
//! policy... except the use of a second chance strategy.
use filetime::FileTime;
use std::fs::DirEntry;
use std::io::ErrorKind;
use std::io::Result;
use std::path::Path;
use std::path::PathBuf;

use crate::second_chance;

/// A `CachedFile` represents what we know about a given file in a raw
/// cache directory: its direntry, mtime, and atime.
struct CachedFile {
    entry: DirEntry,
    mtime: FileTime,
    accessed: bool,
}

/// Removes a file if it exists.
fn ensure_file_removed(path: &Path) -> Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(()),
        err => err,
    }
}

/// Moves the file at `path` to the back of the second chance list.
fn move_to_back_of_list(path: &Path) -> Result<()> {
    filetime::set_file_mtime(path, FileTime::now())
}

/// Marks the file at `path` as read-only.
fn set_read_only(path: &Path) -> Result<()> {
    let mut permissions = std::fs::symlink_metadata(path)?.permissions();

    permissions.set_readonly(true);
    std::fs::set_permissions(path, permissions)
}

/// Sets the access bit to true for the file at `path`: the next time
/// that file is up for eviction, it will get a second chance.  Returns
/// true if the file was found, false otherwise.
///
/// In most cases, there is no need to explicitly call this function:
/// the operating system will automatically perform the required
/// update while opening the file at `path`.
pub fn touch(path: impl AsRef<Path>) -> Result<bool> {
    fn run(path: &Path) -> Result<bool> {
        match filetime::set_file_atime(path, FileTime::now()) {
            Ok(()) => Ok(true),
            // It's OK if the file we're trying to touch was removed:
            // things do disappear from caches.
            Err(e) if e.kind() == ErrorKind::NotFound => Ok(false),
            Err(e) => Err(e),
        }
    }

    run(path.as_ref())
}

/// Consumes the file `from` and publishes it to the raw cache file
/// `to`, a file directly under the raw cache directory.
///
/// On success, `from` is deleted.  On failure, the cache directory is
/// always in a valid state.
///
/// The rename will be atomic, but the source file must be private to
/// the caller: this function accesses the source file multiple time.
///
/// If durability is necessary, the caller is responsible for
/// `sync_data`ing the contents of `from`.  This function does not
/// fsync the cache directory itself: it's a cache, so stale contents
/// are assumed safe.
pub fn insert_or_update(from: impl AsRef<Path>, to: impl AsRef<Path>) -> Result<()> {
    fn run(from: &Path, to: &Path) -> Result<()> {
        // Move to the back of the list before publishing: if a reader
        // comes in right away, we want it to set the access bit.
        move_to_back_of_list(from)?;
        set_read_only(from)?;
        std::fs::rename(from, to)?;
        ensure_file_removed(from)
    }

    run(from.as_ref(), to.as_ref())
}

/// Consumes the file `from` and publishes it to the raw cache file
/// `to`, a file directly under the raw cache directly, only if that
/// file does not already exist.
///
/// If the file exists, `from` is still consumed, but `to` is not
/// updated; instead, its access bit is set to true.
///
/// On success, `from` is deleted.  On failure, the cache directory is
/// always in a valid state.
///
/// The link will be atomic, but the source file must be private to
/// the caller: this function accesses the source file multiple times.
///
/// If durability is necessary, the caller is responsible for
/// `sync_data`ing the contents of `from`.  This function does not
/// fsync the cache directory itself: it's a cache, so stale contents
/// are assumed safe.
pub fn insert_or_touch(from: impl AsRef<Path>, to: impl AsRef<Path>) -> Result<()> {
    fn run(from: &Path, to: &Path) -> Result<()> {
        // Optimise for the successful publication case: we expect callers
        // to only `insert_or_touch` after a failed lookup, so the `link`
        // call will only fail with EEXIST if another writer raced with us.
        move_to_back_of_list(from)?;
        set_read_only(from)?;
        match std::fs::hard_link(from, to) {
            Ok(()) => {}
            // The destination file already exists; we just have to mark
            // it as accessed.
            Err(e) if e.kind() == ErrorKind::AlreadyExists => {
                touch(to)?;
            }
            err => err?,
        }

        ensure_file_removed(from)
    }

    run(from.as_ref(), to.as_ref())
}

impl second_chance::Entry for CachedFile {
    type Rank = FileTime;

    #[inline]
    fn rank(&self) -> FileTime {
        self.mtime
    }

    #[inline]
    fn accessed(&self) -> bool {
        self.accessed
    }
}

impl CachedFile {
    fn new(entry: DirEntry, meta: &std::fs::Metadata) -> CachedFile {
        let atime = FileTime::from_last_access_time(meta);
        let mtime = FileTime::from_last_modification_time(meta);

        CachedFile {
            entry,
            mtime,
            accessed: atime > mtime,
        }
    }
}

/// Attempts to list all the cached files directly under `cache_dir`.
///
/// On success, returns the list of cached files and an estimate of
/// the number of files under `cache_dir`.
///
/// That estimate is always greater than or equal to the number of
/// `CachedFile`s returned in the vector.
fn collect_cached_files(cache_dir: &Path) -> Result<(Vec<CachedFile>, u64)> {
    let mut cache = Vec::new();
    let mut count = 0;
    for maybe_entry in std::fs::read_dir(cache_dir)? {
        count += 1;
        if let Ok(entry) = maybe_entry {
            let meta = match entry.metadata() {
                Ok(meta) => meta,
                Err(e) if e.kind() == ErrorKind::NotFound => continue,
                err => err?,
            };

            if meta.is_dir() {
                // Don't count subdirectories, we never delete them.
                count -= 1;
            } else {
                cache.push(CachedFile::new(entry, &meta));
            }
        }
    }

    // We increment before pushing to `cache`.
    assert!(count >= cache.len() as u64);
    Ok((cache, count))
}

/// Applies the `update` to files in `parent`.
fn apply_update(parent: PathBuf, update: second_chance::Update<CachedFile>) -> Result<()> {
    // When the `parent` path is long, we can spend most of our time
    // constructing paths with `DirEntry::path`.  Directly push/pop
    // after the parent directory instead.
    let mut cached = parent;

    for entry in update.to_evict {
        cached.push(entry.entry.file_name());
        ensure_file_removed(&cached)?;
        cached.pop();
    }

    for entry in update.to_move_back {
        cached.push(entry.entry.file_name());
        match move_to_back_of_list(&cached) {
            Ok(()) => {}
            // Silently ignore ENOENT: things do disappear from caches.
            Err(e) if e.kind() == ErrorKind::NotFound => {}
            err => err?,
        }
        cached.pop();
    }

    Ok(())
}

/// Attempts to prune the contents of `cache_dir` down to at most
/// `capacity` files, with a second chance policy.
///
/// On success, returns an estimate of the number of files remaining
/// in `cache_dir` and the number of files deleted.
///
/// Ignores any subdirectory of `cache_dir`.
pub fn prune(cache_dir: PathBuf, capacity: usize) -> Result<(u64, usize)> {
    let (cached_files, count) = collect_cached_files(&cache_dir)?;
    let update = second_chance::Update::new(cached_files, capacity);
    let num_evicted = update.to_evict.len();

    // `CachedFile` doesn't implement `Clone` (nor can `Update::new`
    // assume the trait is available), so the values in `to_evict`
    // must all come from `cached_files`; since `cached_files.len() <=
    // count`, we must find `num_evicted <= count`.
    assert!(num_evicted as u64 <= count);

    apply_update(cache_dir, update)?;
    Ok((count - (num_evicted as u64), num_evicted))
}

/// Removing a file should remove that file.
#[test]
fn test_remove_file() {
    use test_dir::{DirBuilder, FileType, TestDir};
    let temp = TestDir::temp().create("cache_file", FileType::ZeroFile(10));

    let path = temp.path("cache_file");
    assert!(std::fs::metadata(&path).is_ok());
    assert!(ensure_file_removed(&path).is_ok());

    assert!(matches!(std::fs::metadata(&path),
                     Err(e) if e.kind() == ErrorKind::NotFound));

    // Removing a file that does not exist is ok.
    assert!(ensure_file_removed(&path).is_ok());
    assert!(matches!(std::fs::metadata(&path),
                     Err(e) if e.kind() == ErrorKind::NotFound));
}

/// ensure_file_removed should succeed if the file is already gone.
#[test]
fn test_remove_inexistent_file() {
    use test_dir::{DirBuilder, TestDir};
    let temp = TestDir::temp();

    let path = temp.path("cache_file");
    assert!(matches!(std::fs::metadata(&path),
                     Err(e) if e.kind() == ErrorKind::NotFound));

    // Removing a file that does not exist is ok.
    assert!(ensure_file_removed(&path).is_ok());
    assert!(matches!(std::fs::metadata(&path),
                     Err(e) if e.kind() == ErrorKind::NotFound));
}

/// Ensures enough time has elapsed for atime / mtime to change: some
/// file systems only work at second granularity.
#[cfg(test)]
fn advance_time() {
    std::thread::sleep(std::time::Duration::from_secs_f64(1.5));
}

/// Moving to the back of the list should increase the file's rank and
/// not set the accessed bit.
#[test]
fn test_back_of_list() {
    use crate::second_chance::Entry;
    use test_dir::{DirBuilder, FileType, TestDir};

    let temp = TestDir::temp().create("old_cache_file", FileType::ZeroFile(10));

    let path = temp.path("old_cache_file");

    let get_entry = || {
        let (mut files, count) =
            collect_cached_files(path.parent().unwrap()).expect("directory listing must succeed");
        assert_eq!(count, 1);
        assert_eq!(files.len(), 1);

        files.pop().expect("vec is non-empty")
    };

    let old_entry = get_entry();
    advance_time();
    move_to_back_of_list(&path).expect("call should succeed");
    let new_entry = get_entry();

    assert!(new_entry.rank() > old_entry.rank());
    assert!(!new_entry.accessed());
}

/// Setting a file read only should... make it read only.
#[test]
fn test_set_read_only() {
    use test_dir::{DirBuilder, FileType, TestDir};

    let temp = TestDir::temp().create("old_cache_file", FileType::ZeroFile(10));

    let path = temp.path("old_cache_file");

    // The file should be initially read-write
    {
        let permissions = std::fs::metadata(&path)
            .expect("metadata should succeed")
            .permissions();

        assert!(!permissions.readonly());
    }

    set_read_only(&path).expect("set_read_only should succeed");

    // The file should now be read-only.
    {
        let permissions = std::fs::metadata(&path)
            .expect("metadata should succeed")
            .permissions();

        assert!(permissions.readonly());
    }
}

/// Touching should set the accessed bit, but not change the rank.
#[test]
fn test_touch() {
    use crate::second_chance::Entry;
    use test_dir::{DirBuilder, FileType, TestDir};

    let temp = TestDir::temp().create("old_cache_file", FileType::ZeroFile(10));

    let path = temp.path("old_cache_file");

    let get_entry = || {
        let (mut files, count) =
            collect_cached_files(path.parent().unwrap()).expect("directory listing must succeed");
        assert_eq!(count, 1);
        assert_eq!(files.len(), 1);

        files.pop().expect("vec is non-empty")
    };

    move_to_back_of_list(&path).expect("call should succeed");
    let old_entry = get_entry();
    assert!(!old_entry.accessed());

    advance_time();
    // Should return true: the file exists.
    assert!(touch(&path).expect("call should succeed"));
    let new_entry = get_entry();

    assert_eq!(new_entry.rank(), old_entry.rank());
    assert!(new_entry.accessed());
}

/// Touching should set the accessed bit, but not change the rank.
#[test]
fn test_touch_missing() {
    use test_dir::{DirBuilder, TestDir};

    let temp = TestDir::temp();
    // Should return file: the file does not exist.
    assert!(!touch(&temp.path("absent")).expect("should succeed on missing files"));
}

/// Reading a file should set the accessed bit, but not change the rank.
#[test]
fn test_touch_by_open() {
    use crate::second_chance::Entry;
    use test_dir::{DirBuilder, FileType, TestDir};

    let temp = TestDir::temp().create("old_cache_file", FileType::ZeroFile(10));

    let path = temp.path("old_cache_file");

    let get_entry = || {
        let (mut files, count) =
            collect_cached_files(path.parent().unwrap()).expect("directory listing must succeed");
        assert_eq!(count, 1);
        assert_eq!(files.len(), 1);

        files.pop().expect("vec is non-empty")
    };

    move_to_back_of_list(&path).expect("call should succeed");
    let old_entry = get_entry();
    assert!(!old_entry.accessed());

    advance_time();
    let _ = std::fs::read(&path).expect("read should succeed");
    let new_entry = get_entry();

    assert_eq!(new_entry.rank(), old_entry.rank());
    assert!(new_entry.accessed());
}

/// Moving to the back of the list should clear the access bit.
#[test]
fn test_back_of_list_after_touch() {
    use crate::second_chance::Entry;
    use test_dir::{DirBuilder, FileType, TestDir};

    let temp = TestDir::temp().create("old_cache_file", FileType::ZeroFile(10));

    let path = temp.path("old_cache_file");

    let get_entry = || {
        let (mut files, count) =
            collect_cached_files(path.parent().unwrap()).expect("directory listing must succeed");
        assert_eq!(count, 1);
        assert_eq!(files.len(), 1);

        files.pop().expect("vec is non-empty")
    };

    advance_time();
    touch(&path).expect("call should succeed");
    let old_entry = get_entry();
    assert!(old_entry.accessed());

    advance_time();
    move_to_back_of_list(&path).expect("call should succeed");
    let new_entry = get_entry();

    assert!(new_entry.rank() > old_entry.rank());
    assert!(!new_entry.accessed());
}

/// Moving to the back of the list should clear the access bit, even when
/// set by opening the file for read.
#[test]
fn test_back_of_list_after_open() {
    use crate::second_chance::Entry;
    use test_dir::{DirBuilder, FileType, TestDir};

    let temp = TestDir::temp().create("old_cache_file", FileType::ZeroFile(10));

    let path = temp.path("old_cache_file");

    let get_entry = || {
        let (mut files, count) =
            collect_cached_files(path.parent().unwrap()).expect("directory listing must succeed");
        assert_eq!(count, 1);
        assert_eq!(files.len(), 1);

        files.pop().expect("vec is non-empty")
    };

    advance_time();
    let _ = std::fs::read(&path).expect("read should succeed");
    let old_entry = get_entry();
    assert!(old_entry.accessed());

    advance_time();
    move_to_back_of_list(&path).expect("call should succeed");
    let new_entry = get_entry();

    assert!(new_entry.rank() > old_entry.rank());
    assert!(!new_entry.accessed());
}

/// Inserting a new file should move to the destination, and mark the
/// file as not yet accessed.
#[test]
fn test_insert_empty() {
    use crate::second_chance::Entry;
    use test_dir::{DirBuilder, FileType, TestDir};

    let temp = TestDir::temp().create("temp_file", FileType::RandomFile(10));

    let path = temp.path("temp_file");
    let dst = temp.path("cache");

    let get_entry = || {
        let (mut files, count) =
            collect_cached_files(path.parent().unwrap()).expect("directory listing must succeed");
        assert_eq!(count, 1);
        assert_eq!(files.len(), 1);

        files.pop().expect("vec is non-empty")
    };

    // Read *after* the file is created, to make sure it would have
    // the accessed bit set.
    advance_time();
    let payload = std::fs::read(&path).expect("read should succeed");

    insert_or_update(&path, &dst).expect("insert_or_touch should succeed");
    // The old file should be gone.
    assert!(matches!(std::fs::metadata(&path),
                     Err(e) if e.kind() == ErrorKind::NotFound));

    // The destination file should not be marked as accessed.
    assert!(!get_entry().accessed());
    // The destination file should have the original payload.
    assert_eq!(&payload, &std::fs::read(&dst).expect("read should succeed"));
    // The destination file should now be read-only.
    assert!(std::fs::metadata(&dst)
        .expect("metadata should succeed")
        .permissions()
        .readonly());
}

/// Inserting a new file over an old one should overwrite, and mark
/// the file as not yet accessed.
#[test]
fn test_insert_overwrite() {
    use crate::second_chance::Entry;
    use test_dir::{DirBuilder, FileType, TestDir};

    let temp = TestDir::temp()
        .create("temp_file", FileType::RandomFile(10))
        .create("cache", FileType::ZeroFile(100));

    let path = temp.path("temp_file");
    let dst = temp.path("cache");

    let get_entry = || {
        let (mut files, count) =
            collect_cached_files(path.parent().unwrap()).expect("directory listing must succeed");
        assert_eq!(count, 1);
        assert_eq!(files.len(), 1);

        files.pop().expect("vec is non-empty")
    };

    // Read *after* the file is created, to make sure it would have
    // the accessed bit set.
    advance_time();
    let payload = std::fs::read(&path).expect("read should succeed");

    insert_or_update(&path, &dst).expect("insert_or_touch should succeed");
    // The old file should be gone.
    assert!(matches!(std::fs::metadata(&path),
                     Err(e) if e.kind() == ErrorKind::NotFound));

    // The destination file should not be marked as accessed.
    assert!(!get_entry().accessed());
    // The destination file should have the original payload.
    assert_eq!(&payload, &std::fs::read(&dst).expect("read should succeed"));
}

/// Inserting a new file should move to the destination, and mark the
/// file as not yet accessed.
#[test]
fn test_insert_or_touch_empty() {
    use crate::second_chance::Entry;
    use test_dir::{DirBuilder, FileType, TestDir};

    let temp = TestDir::temp().create("temp_file", FileType::RandomFile(10));

    let path = temp.path("temp_file");
    let dst = temp.path("cache");

    let get_entry = || {
        let (mut files, count) =
            collect_cached_files(path.parent().unwrap()).expect("directory listing must succeed");
        assert_eq!(count, 1);
        assert_eq!(files.len(), 1);

        files.pop().expect("vec is non-empty")
    };

    // Read *after* the file is created, to make sure it would have
    // the accessed bit set.
    advance_time();
    let payload = std::fs::read(&path).expect("read should succeed");

    insert_or_touch(&path, &dst).expect("insert_or_touch should succeed");
    // The old file should be gone.
    assert!(matches!(std::fs::metadata(&path),
                     Err(e) if e.kind() == ErrorKind::NotFound));

    // The destination file should not be marked as accessed.
    assert!(!get_entry().accessed());
    // The destination file should have the original payload.
    assert_eq!(&payload, &std::fs::read(&dst).expect("read should succeed"));
    // The destination file should now be read-only.
    assert!(std::fs::metadata(&dst)
        .expect("metadata should succeed")
        .permissions()
        .readonly());
}

/// insert_or_touch'ing over an old file should not overwrite, but
/// mark the file as newly accessed.
#[test]
fn test_insert_touch_overwrite() {
    use crate::second_chance::Entry;
    use test_dir::{DirBuilder, FileType, TestDir};

    let temp = TestDir::temp()
        .create("temp_file", FileType::RandomFile(10))
        .create("cache", FileType::RandomFile(10));

    let path = temp.path("temp_file");
    let dst = temp.path("cache");

    let get_entry = || {
        let (mut files, count) =
            collect_cached_files(path.parent().unwrap()).expect("directory listing must succeed");
        assert_eq!(count, 1);
        assert_eq!(files.len(), 1);

        files.pop().expect("vec is non-empty")
    };

    let payload = std::fs::read(&dst).expect("read should succeed");
    // Clear the access bit.
    move_to_back_of_list(&dst).expect("move to back should succeed");

    advance_time();

    insert_or_touch(&path, &dst).expect("insert_or_touch should succeed");
    // The old file should be gone.
    assert!(matches!(std::fs::metadata(&path),
                     Err(e) if e.kind() == ErrorKind::NotFound));

    // The destination file should be marked as accessed.
    assert!(get_entry().accessed());
    // The destination file should have the original payload.
    assert_eq!(&payload, &std::fs::read(&dst).expect("read should succeed"));
}

/// Smoke test that we find the order we expect.
#[test]
fn test_collect_cached_files() {
    use crate::second_chance::Entry;
    use std::ffi::OsString;
    use test_dir::{DirBuilder, FileType, TestDir};

    let temp = TestDir::temp()
        .create("a", FileType::RandomFile(10))
        .create("b", FileType::RandomFile(10))
        .create("c", FileType::RandomFile(10))
        .create("d", FileType::RandomFile(10))
        // The directory should be ignored.
        .create("a_directory", FileType::Dir);

    // Set up the order: d, a, c, b
    advance_time();
    move_to_back_of_list(&temp.path("d")).expect("should succeed");
    advance_time();
    move_to_back_of_list(&temp.path("a")).expect("should succeed");
    advance_time();
    move_to_back_of_list(&temp.path("c")).expect("should succeed");
    advance_time();
    move_to_back_of_list(&temp.path("b")).expect("should succeed");

    // Touch d and c.
    advance_time();
    touch(&temp.path("d")).expect("should succeed");
    touch(&temp.path("c")).expect("should succeed");

    let (mut cached, count) = collect_cached_files(&temp.path(".")).expect("should succeed");

    assert_eq!(count, 4);
    cached.sort_by_key(|e| e.rank());

    assert_eq!(
        cached
            .iter()
            .map(|e| e.entry.file_name())
            .collect::<Vec<OsString>>(),
        vec!["d", "a", "c", "b"]
    );

    assert_eq!(
        cached.iter().map(|e| e.accessed()).collect::<Vec<bool>>(),
        vec![true, false, true, false]
    );
}

/// Similar setup, but now directly delete / move files.
#[test]
fn test_apply_update() {
    use crate::second_chance::Entry;
    use std::ffi::OsString;
    use test_dir::{DirBuilder, FileType, TestDir};

    let temp = TestDir::temp()
        .create("a", FileType::RandomFile(10))
        .create("b", FileType::RandomFile(10))
        .create("c", FileType::RandomFile(10))
        .create("d", FileType::RandomFile(10))
        // We will add these files to the update before deleting them;
        // `apply_update` shouldn't blow up.
        .create("deleted_touch", FileType::RandomFile(10))
        .create("already_deleted", FileType::RandomFile(10));

    // Set up the order: d, a, c, b
    advance_time();
    move_to_back_of_list(&temp.path("d")).expect("should succeed");
    advance_time();
    move_to_back_of_list(&temp.path("a")).expect("should succeed");
    advance_time();
    move_to_back_of_list(&temp.path("c")).expect("should succeed");
    advance_time();
    move_to_back_of_list(&temp.path("b")).expect("should succeed");

    // Move deleted_touch and already_deleted to the back.
    advance_time();
    move_to_back_of_list(&temp.path("deleted_touch")).expect("should succeed");
    advance_time();
    move_to_back_of_list(&temp.path("already_deleted")).expect("should succeed");

    // Touch d and c.
    advance_time();
    touch(&temp.path("d")).expect("should succeed");
    touch(&temp.path("c")).expect("should succeed");

    // move d to the back, delete a.
    {
        let (mut cached, count) = collect_cached_files(&temp.path(".")).expect("should succeed");

        assert_eq!(count, 6);
        cached.sort_by_key(|e| e.rank());

        assert_eq!(
            cached
                .iter()
                .map(|e| e.entry.file_name())
                .collect::<Vec<OsString>>(),
            vec!["d", "a", "c", "b", "deleted_touch", "already_deleted"]
        );

        let mut to_evict = Vec::new();
        let mut to_move_back = Vec::new();

        to_move_back.extend(cached.drain(0..1));
        to_evict.extend(cached.drain(0..1));
        to_evict.push(cached.pop().expect("is non-empty"));
        to_move_back.push(cached.pop().expect("is non-empty"));

        // Delete these files before applying the update; it should
        // just ignore their entries.
        std::fs::remove_file(&temp.path("deleted_touch")).expect("deletion should succeed");
        std::fs::remove_file(&temp.path("already_deleted")).expect("deletion should succeed");

        apply_update(
            temp.path("."),
            second_chance::Update {
                to_evict,
                to_move_back,
            },
        )
        .expect("update should succeed");
    }

    let (mut cached, count) = collect_cached_files(&temp.path(".")).expect("should succeed");

    assert_eq!(count, 3);
    cached.sort_by_key(|e| e.rank());

    // We should find 3 files, in this order
    assert_eq!(
        cached
            .iter()
            .map(|e| e.entry.file_name())
            .collect::<Vec<OsString>>(),
        vec!["c", "b", "d"]
    );

    // And "d"'s accessed bit should be reset.
    assert_eq!(
        cached.iter().map(|e| e.accessed()).collect::<Vec<bool>>(),
        vec![true, false, false]
    );
}

/// Same setup, but now directly delete / move files.
#[test]
fn test_prune() {
    use crate::second_chance::Entry;
    use std::ffi::OsString;
    use test_dir::{DirBuilder, FileType, TestDir};

    let temp = TestDir::temp()
        .create("a", FileType::RandomFile(10))
        .create("b", FileType::RandomFile(10))
        .create("c", FileType::RandomFile(10))
        .create("d", FileType::RandomFile(10));

    // Set up the order: d, a, c, b
    advance_time();
    move_to_back_of_list(&temp.path("d")).expect("should succeed");
    advance_time();
    move_to_back_of_list(&temp.path("a")).expect("should succeed");
    advance_time();
    move_to_back_of_list(&temp.path("c")).expect("should succeed");
    advance_time();
    move_to_back_of_list(&temp.path("b")).expect("should succeed");

    // Touch d and c.
    advance_time();
    touch(&temp.path("d")).expect("should succeed");
    touch(&temp.path("c")).expect("should succeed");

    // With capacity for 3 files, we should move "d" to the back,
    // and delete "a".
    //
    // That leaves us with 3 files (1 deletion).
    assert_eq!(
        prune(temp.path("."), 3).expect("prune should succeed"),
        (3, 1)
    );

    let (mut cached, count) = collect_cached_files(&temp.path(".")).expect("should succeed");

    assert_eq!(count, 3);
    cached.sort_by_key(|e| e.rank());

    // We should find 3 files, in this order
    assert_eq!(
        cached
            .iter()
            .map(|e| e.entry.file_name())
            .collect::<Vec<OsString>>(),
        vec!["c", "b", "d"]
    );

    // And "d"'s accessed bit should be reset.
    assert_eq!(
        cached.iter().map(|e| e.accessed()).collect::<Vec<bool>>(),
        vec![true, false, false]
    );
}
