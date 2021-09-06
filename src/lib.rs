mod cache_dir;
mod plain;
pub mod raw_cache;
mod readonly;
pub mod second_chance;
mod sharded;
mod trigger;

pub use plain::PlainCache;
pub use readonly::ReadOnlyCache;
pub use readonly::ReadOnlyCacheBuilder;
pub use sharded::ShardedCache;

/// Sharded cache keys consist of a filename and two hash values.  The
/// two hashes should be computed by distinct functions of the key's
/// name, and each hash function must be identical for all processes
/// that access the same sharded cache directory.
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
