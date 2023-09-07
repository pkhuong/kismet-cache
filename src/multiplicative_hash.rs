/// Multiplicative hash structs implement
/// [Dietzfelbinger's universal multiplicative hash function](https://link.springer.com/chapter/10.1007/978-3-319-98355-4_15)
/// with `const fn` keyed constructors, and pair that with a range
/// reduction function from `u64` to a `usize` range that extends
/// Dietzfelbinger's power-of-two scheme.
///
/// For a truly faithful implementation of Dietzfelbinger's
/// multiply-add-shift for 64-bit domain and range, we'd want 128-bit
/// multiplier and addend.  This 64-bit version is faster and good
/// enough for the cases we care about: we shouldn't need that many
/// bits to assign a cache shard, and the multiplicative hash is only
/// used as a last resort mixer to avoid really bad behaviour when the
/// `Key`s' hashes are clusters.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct MultiplicativeHash {
    // Pseudorandom odd multiplier
    multiplier: u64,
    // Pseudorandom value added to the product
    addend: u64,
}

/// Maps values in `[0, u64::MAX]` to `[0, domain)` linearly.
///
/// As a special case, this function returns 0 instead of erroring out
/// when `domain == 0`.
#[inline(always)]
const fn reduce(x: u64, domain: usize) -> usize {
    ((domain as u128 * x as u128) >> 64) as usize
}

impl MultiplicativeHash {
    /// Constructs a `MultiplicativeHash` with the arguments as parameters.
    /// The multiplier is converted to an odd integer if necessary.
    pub const fn new(multiplier: u64, addend: u64) -> MultiplicativeHash {
        MultiplicativeHash {
            multiplier: multiplier | 1,
            addend,
        }
    }

    /// Deterministically constructs a `MultiplicativeHash` with
    /// parameters derived from the `key`, wih a SHA-256 hash.
    pub const fn new_keyed(key: &[u8]) -> MultiplicativeHash {
        use extendhash::sha256;

        let hash = sha256::compute_hash(key);
        let multiplier = [
            hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7],
        ];
        let addend = [
            hash[8], hash[9], hash[10], hash[11], hash[12], hash[13], hash[14], hash[15],
        ];

        MultiplicativeHash::new(u64::from_le_bytes(multiplier), u64::from_le_bytes(addend))
    }

    /// Constructs a new pseudorandom `MultiplicativeHash`.
    #[cfg(test)]
    pub fn new_random() -> MultiplicativeHash {
        use rand::Rng;

        let mut rnd = rand::thread_rng();
        MultiplicativeHash::new(rnd.gen(), rnd.gen())
    }

    /// Mixes `value` with this hash's parameters.  If you must
    /// truncate the result, use its high bits.
    #[inline(always)]
    pub const fn mix(&self, value: u64) -> u64 {
        value
            .wrapping_mul(self.multiplier)
            .wrapping_add(self.addend)
    }

    /// Mixes `value` and maps the result to a usize less than range.
    ///
    /// If `range == 0`, always returns 0.
    #[inline(always)]
    pub const fn map(&self, value: u64, range: usize) -> usize {
        reduce(self.mix(value), range)
    }
}

/// Smoke test the `reduce` function.
#[test]
fn test_reduce() {
    // Mapping to an empty range should always return 0.
    assert_eq!(reduce(0, 0), 0);
    assert_eq!(reduce(u64::MAX, 0), 0);

    // Smoke test the range reduction
    assert_eq!(reduce(0, 17), 0);
    assert_eq!(reduce(u64::MAX / 17, 17), 0);
    assert_eq!(reduce(1 + u64::MAX / 17, 17), 1);
    assert_eq!(reduce(u64::MAX, 17), 16);
}

/// Mapping to a power-of-two sized range is the same as taking the
/// high bits.
#[test]
fn test_reduce_power_of_two() {
    assert_eq!(reduce(10 << 33, 1 << 32), 10 << 1);
    assert_eq!(reduce(15 << 60, 1 << 8), 15 << 4);
}

/// Construct two different hashers.  We should get different values
/// for `mix`.
#[test]
fn test_mix() {
    let h1 = MultiplicativeHash::new_keyed(b"h1");
    let h2 = MultiplicativeHash::new_keyed(b"h2");

    assert!(h1 != h2);

    assert!(h1.mix(0) != h2.mix(0));
    assert!(h1.mix(1) != h2.mix(1));
    assert!(h1.mix(42) != h2.mix(42));
    assert!(h1.mix(u64::MAX) != h2.mix(u64::MAX));
}

/// Construct two random hashers.  We should get different values
/// for `mix`.
#[test]
fn test_random_mix() {
    let h1 = MultiplicativeHash::new_random();
    let h2 = MultiplicativeHash::new_random();

    assert!(h1 != h2);

    assert!(h1.mix(0) != h2.mix(0));
    assert!(h1.mix(1) != h2.mix(1));
    assert!(h1.mix(42) != h2.mix(42));
    assert!(h1.mix(u64::MAX) != h2.mix(u64::MAX));
}

/// Construct two different hashers.  We should get different
/// values for `map`.
#[test]
fn test_map() {
    let h1 = MultiplicativeHash::new_keyed(b"h1");
    let h2 = MultiplicativeHash::new_keyed(b"h2");

    assert!(h1 != h2);

    assert!(h1.map(0, 1024) != h2.map(0, 1024));
    assert!(h1.map(1, 1234) != h2.map(1, 1234));
    assert!(h1.map(42, 4567) != h2.map(42, 4567));
    assert!(h1.map(u64::MAX, 789) != h2.map(u64::MAX, 789));
}

/// Confirm that construction is const and deterministic.
#[test]
fn test_new_keyed() {
    const H: MultiplicativeHash = MultiplicativeHash::new_keyed(b"asdfg");

    // Given the nature of the hash function, two points suffice to
    // derive the parameters.

    // addend = 7162733811001658625
    assert_eq!(H.mix(0), 7162733811001658625);
    assert_eq!(H.addend, 7162733811001658625);
    // multiplier = 14551484392748644090 - addend = 7388750581746985465
    assert_eq!(H.mix(1), 14551484392748644090);
    assert_eq!(H.multiplier, 7388750581746985465);

    assert_eq!(
        H,
        MultiplicativeHash::new(7388750581746985465, 7162733811001658625)
    );

    // But it doesn't hurt to test a couple more points.
    assert_eq!(
        H.mix(42),
        42u64
            .wrapping_mul(7388750581746985465)
            .wrapping_add(7162733811001658625)
    );
    assert_eq!(
        H.mix(u64::MAX),
        u64::MAX
            .wrapping_mul(7388750581746985465)
            .wrapping_add(7162733811001658625)
    );
}
