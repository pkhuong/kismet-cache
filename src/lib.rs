//! Kismet implements multiprocess lock-free[^lock-free-fs]
//! crash-safe and (roughly) bounded persistent caches stored
//! in filesystem directories, with a
//! [Second Chance](https://en.wikipedia.org/wiki/Page_replacement_algorithm#Second-chance)
//! eviction strategy.  The maintenance logic is batched and invoked
//! at periodic jittered intervals to make sure accesses amortise to a
//! constant number of filesystem system calls and logarithmic (in the
//! number of cached file) time complexity.  That's good for performance,
//! and enables lock-freedom,[^unlike-ccache] but does mean that
//! caches are expected to temporarily grow past their capacity
//! limits, although rarely by more than a factor of 2 or 3.
//!
//! [^lock-free-fs]: Inasmuch as anything that makes a lot of syscalls
//! can be "lock-free."  The cache access algorithms implement a
//! protocol that makes a bounded number of file open, rename, or link
//! syscalls; in other words, reads and writes are as wait-free as
//! these syscalls.  The batched second-chance maintenance algorithm,
//! on the other hand, is merely lock-free: it could theoretically get
//! stuck if writers kept adding new files to a cache (sub)directory.
//! Again, this guarantee is in terms of file access and directory
//! enumeration syscalls, and maintenance is only as lock-free as the
//! underlying syscalls.  However, we can assume the kernel is
//! reasonably well designed, and doesn't let any sequence of syscalls
//! keep its hand on kernel locks forever.
//!
//! [^unlike-ccache]: This design choice is different from, e.g.,
//! [ccache](https://ccache.dev/)'s, which attempts to maintain
//! statistics per shard with locked files.  Under high load, the lock
//! to update ccache statistics becomes a bottleneck.  Yet, despite
//! taking this hit, ccache batches evictions like Kismet, because
//! cleaning up a directory is slow; up-to-date access statistics
//! aren't enough to enforce tight cache capacity limits.
//!
//! In addition to constant per-cache space overhead, each Kismet
//! cache maintains a variable-length [`std::path::PathBuf`] for the
//! directory, one byte of lock-free metadata per shard, and no other
//! non-heap resource (i.e., Kismet caches do not hold on to file
//! objects).  This holds for individual cache directories; when
//! stacking multiple caches in a [`Cache`] or [`ReadOnlyCache`], the
//! read-write cache and all constituent read-only caches will each
//! have their own `PathBuf` and per-shard metadata.
//!
//! When a Kismet cache triggers second chance evictions, it will
//! allocate temporary data.  That data's size is proportional to the
//! number of files in the cache shard subdirectory undergoing
//! eviction (or the whole directory for a plain unsharded cache), and
//! includes a copy of the basename (without the path prefix) for each
//! cached file in the subdirectory (or plain cache directory).  This
//! eviction process is linearithmic-time in the number of files in
//! the cache subdirectory (directory), and is invoked periodically,
//! so as to amortise the maintenance overhead to logarithmic (in the
//! total number of files in the subdirectory) time per write to a
//! cache subdirectory, and constant file operations per write.
//!
//! Kismet does not pre-allocate any long-lived file object, so it may
//! need to temporarily open file objects.  Each call nevertheless
//! bounds the number of concurrently allocated file objects; the
//! current logic never allocates more than two concurrent file
//! objects.
//!
//! The load (number of files) in each cache may exceed the cache's
//! capacity because there is no centralised accounting, except for
//! what filesystems provide natively.  This design choice forces
//! Kismet to amortise maintenance calls with randomisation, but also
//! means that any number of threads or processes may safely access
//! the same cache directories without any explicit synchronisation.
//!
//! Filesystems can't be trusted to provide much; Kismet only relies
//! on file modification times (`mtime`), and on file access times
//! (`atime`) that are either less than or equal to the `mtime`, or
//! greater than the `mtime` (i.e., `relatime` is acceptable).  This
//! implies that cached files should not be linked in multiple Kismet
//! cache directories.  It is however safe to hardlink cached files in
//! multiple places, as long as the files are not modified, or their
//! `mtime` otherwise updated, through these non-Kismet links.
//!
//! # Plain and sharded caches
//!
//! Kismet cache directories are plain (unsharded) or sharded.
//!
//! Plain Kismet caches are simply directories where the cache entry for
//! "key" is the file named "key."  These are most effective for
//! read-only access to cache directories managed by some other
//! process, or for small caches of up to ~100 cached files.
//!
//! Sharded caches scale to higher capacities, by indexing into one of
//! a constant number of shard subdirectories with a hash, and letting
//! each shard manage fewer files (ideally 10-100 files).  They are
//! also much less likely to grow to multiples of the target capacity
//! than plain (unsharded) cache directories.
//!
//! Simple usage should be covered by the [`ReadOnlyCache`] or
//! [`Cache`] structs, which wrap [`plain::Cache`] and
//! [`sharded::Cache`] in a convenient type-erased interface.
//!
//! While the cache code syncs cached data files by default, it does
//! not sync the parent cache directories: we assume that it's safe,
//! if unfortunate, for caches to lose data or revert to an older
//! state after kernel or hardware crashes.  In general, the code
//! attempts to be robust again direct manipulation of the cache
//! directories.  It's always safe to delete cache files from kismet
//! directories (ideally not recently created files in `.kismet_temp`
//! subdirectories), and even *adding* files should mostly do what one
//! expects: they will be picked up if they're in the correct place
//! (in a plain unsharded cache directory or in the correct shard
//! subdirectory), and eventually evicted if useless or in the wrong
//! shard.
//!
//! It is however essential to only publish files atomically to the
//! cache directories, and it probably never makes sense to modify
//! cached file objects in place.  In fact, Kismet always set files
//! readonly before publishing them to the cache and always returns
//! read-only [`std::fs::File`] objects for cached data.
//!
//! # Sample usage
//!
//! One could access a list of read-only caches with a [`ReadOnlyCache`].
//!
//! ```no_run
//! const NUM_SHARDS: usize = 10;
//!
//! let read_only = kismet_cache::ReadOnlyCacheBuilder::new()
//!     .plain("/tmp/plain_cache")  // Read first here
//!     .sharded("/tmp/sharded_cache", NUM_SHARDS)  // Then try there.
//!     .take()
//!     .build();
//!
//! // Attempt to read the file for key "foo", with primary hash 1
//! // and second hash 2, first from `/tmp/plain.cache`, and then
//! // from `/tmp/sharded_cache`.  In practice, the hashes should
//! // probably be populated with by implementing the `From<&'a T>`
//! // trait, and passing a `&T` to the cache methods.
//! read_only.get(kismet_cache::Key::new("foo", 1, 2));
//! ```
//!
//! Read-write accesses should use a [`Cache`]:
//!
//! ```no_run
//! struct CacheKey {
//!   // ...
//! }
//!
//! fn get_contents(key: &CacheKey) -> Vec<u8> {
//!   // ...
//!   # unreachable!()
//! }
//!
//! impl<'a> From<&'a CacheKey> for kismet_cache::Key<'a> {
//!   fn from(key: &CacheKey) -> kismet_cache::Key {
//!     // ...
//!     # unreachable!()
//!   }
//! }
//!
//!
//! // It's easier to increase the capacity than the number of shards,
//! // so, when in doubt, prefer a few too many shards with a lower
//! // capacity.  It's not incorrect to increase the number of shards,
//! // but will result in lost cached data (eventually deleted), since
//! // Kismet does not assign shards with a consistent hash.
//! const NUM_SHARDS: usize = 100;
//! const CAPACITY: usize = 1000;
//!
//! # fn main() -> std::io::Result<()> {
//! use std::io::Read;
//! use std::io::Write;
//!
//! let cache = kismet_cache::CacheBuilder::new()
//!     .sharded_writer("/tmp/root_cache", NUM_SHARDS, CAPACITY)
//!     .plain_reader("/tmp/extra_cache")  // Try to fill cache misses here
//!     .take()
//!     .build();
//!
//! let key: CacheKey = // ...
//!     # CacheKey {}
//!     ;
//!
//! // Fetches the current cached value for `key`, or populates it with
//! // the closure argument if missing.
//! let mut cached_file = cache
//!     .ensure(&key, |file| file.write_all(&get_contents(&key)))?;
//! let mut contents = Vec::new();
//! cached_file.read_to_end(&mut contents)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Cache directory structure
//!
//! Plain (unsharded) cache directories simply store the value for
//! each `key` under a file named `key`.  They also have a single
//! `.kismet_temp` subdirectory, for temporary files.
//!
//! The second chance algorithm relies on mtime / atime (`relatime`
//! updates suffice), so merely opening a file automatically updates
//! the relevant read tracking metadata.
//!
//! Sharded cache directories store the values for each `key` under
//! one of two shard subdirectories.  The primary and second potential
//! shards are respectively determined by multiplying `Key::hash` and
//! `Key::secondary_hash` by different odd integers before mapping the
//! result to the shard universe with a fixed point scaling.
//!
//! Each subdirectory is named `.kismet_$HEX_SHARD_ID`, and contains
//! cached files with name equal to the cache key, and a
//! `.kismet_temp` subsubdirectory, just like plain unsharded caches.
//! In fact, each such shard is managed exactly like a plain cache.
//!
//! Sharded caches attempt to balance load between two potential
//! shards for each cache key in an attempt to make all shards grow at
//! roughly the same rate.  Once all the shards have reached their
//! capacity, the sharded cache will slowly revert to storing cache
//! keys in their primary shards.
//!
//! This scheme lets plain cache directories easily interoperate with
//! other programs that are not aware of Kismet, and also lets an
//! application use the same directory to back both a plain and a
//! sharded cache (concurrently or not) without any possibility of
//! collision between cached files and Kismet's internal directories.
//!
//! Kismet will always store its internal data in files or directories
//! start start with a `.kismet` prefix, and cached data lives in
//! files with names equal to their keys.  Since Kismet sanitises
//! cache keys to forbid them from starting with `.`, `/`, or `\`, it
//! is always safe for an application to store additional data in
//! files or directories that start with a `.`, as long as they do not
//! collide with the `.kismet` prefix.
mod cache_dir;
mod multiplicative_hash;
pub mod plain;
pub mod raw_cache;
mod readonly;
pub mod second_chance;
pub mod sharded;
mod stack;
mod trigger;

pub use readonly::ReadOnlyCache;
pub use readonly::ReadOnlyCacheBuilder;
pub use stack::Cache;
pub use stack::CacheBuilder;
pub use stack::CacheHit;
pub use stack::CacheHitAction;

/// Kismet cache directories put temporary files under this
/// subdirectory in each cache or cache shard directory.
pub const KISMET_TEMPORARY_SUBDIRECTORY: &str = ".kismet_temp";

/// Cache keys consist of a filename and two hash values.  The two
/// hashes should ideally be computed by distinct functions of the
/// key's name, but Kismet will function correctly if the `hash` and
/// `secondary_hash` are the same.  Each hash function **must** be
/// identical for all processes that access the same sharded cache
/// directory.
///
/// The `name` should not be empty nor start with a dot, forward
/// slash, a backslash: caches will reject any operation on such names
/// with an `ErrorKind::InvalidInput` error.
#[derive(Clone, Copy, Debug)]
pub struct Key<'a> {
    pub name: &'a str,
    pub hash: u64,
    pub secondary_hash: u64,
}

impl<'a> Key<'a> {
    /// Returns a new `Key` for this `name`, `hash`, and `secondary_hash`.
    pub fn new(name: &str, hash: u64, secondary_hash: u64) -> Key {
        Key {
            name,
            hash,
            secondary_hash,
        }
    }
}
