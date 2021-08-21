//! The Second Chance or Clock page replacement policy is a simple
//! approximation of the Least Recently Used policy.  Kismet uses the
//! second chance policy because it can be easily implemented on top
//! of the usual file modification and access times that we can trust
//! operating systems to update for us.
//!
//! This second chance algorithm is optimised for *batch* maintenance:
//! the caller is expected to perform a number of operations
//! (insertions and accesses) on a set of cached entries before
//! calling the linearithmic-time `Update::new()`.

/// The Second Chance implementation works with a set of entries; each
/// entry keeps track of a value that represents its rank in the set,
/// and a single boolean flag, to determine whether the entry has been
/// accessed since the last time it was scanned (and move up in rank).
pub trait Entry {
    /// The Second Chance policy maintains an implicit list of
    /// entries.  Entries in the list are ordered by sorting by Rank;
    /// a lower rank comes earlier in the list of removal victims.
    type Rank: Ord;

    /// Returns the rank value for this entry.
    fn rank(&self) -> Self::Rank;

    /// Returns whether the entry was accessed since it last entered
    /// or was moved back in the list of candidates for removal.
    fn accessed(&self) -> bool;
}

/// An `Update<T>` represents the maintenance operations to perform
/// on a set of Second Chance entries.
pub struct Update<T: Entry> {
    /// List of entries to evict (remove from the cached set)
    pub to_evict: Vec<T>,

    /// List of entries to move back in the list of potential removal
    /// victims, in order (i.e., the first entry in `to_touch` should
    /// be moved directly to the end of the current list, the second
    /// entry right after, etc.).
    pub to_touch: Vec<T>,
}

impl<T: Entry> Update<T> {
    /// Determines how to update a second chance list of `entries`,
    /// so that at most `capacity` entries remain.
    pub fn new(entries: impl IntoIterator<Item = T>, capacity: usize) -> Self {
        let mut sorted_entries: Vec<T> = entries.into_iter().collect();

        if sorted_entries.len() <= capacity {
            return Self {
                to_evict: Vec::new(),
                to_touch: Vec::new(),
            };
        }

        sorted_entries.sort_by_cached_key(|e| e.rank());
        let must_remove = sorted_entries.len() - capacity;
        let mut to_evict = Vec::new();
        let mut to_touch = Vec::new();

        for entry in sorted_entries {
            if to_evict.len() == must_remove {
                break;
            }

            // Give the entry a second chance if has been accessed
            // between the last time it entered the tail of the
            // eviction list and now.
            if entry.accessed() {
                to_touch.push(entry);
            } else {
                // Otherwise, we evict in FIFO order.
                to_evict.push(entry);
            }
        }

        // If we still have elements to remove, we didn't break early
        // from the loop, so every input entry is either in `to_touch`
        // or in `to_evict`.
        if to_evict.len() < must_remove {
            let num_remaining = must_remove - to_evict.len();
            assert!(num_remaining <= to_touch.len());

            let survivors = to_touch.split_off(num_remaining);
            to_evict.extend(to_touch);

            to_touch = survivors;
        }

        Self { to_evict, to_touch }
    }
}

#[cfg(test)]
mod test {
    use crate::second_chance::*;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_derive::Arbitrary;
    use std::collections::HashSet;

    /// A test entry is a u64 timestamp for the virtual time at which
    /// it entered the second chance list, and an "accessed" bool that
    /// is true if the entry has been accessed since it last
    /// (re-)entered the second chance list.
    #[derive(Arbitrary, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
    struct TestEntry(u64, bool);

    impl Entry for TestEntry {
        type Rank = u64;

        fn rank(&self) -> u64 {
            self.0
        }

        fn accessed(&self) -> bool {
            self.1
        }
    }

    /// No-op on an empty list of entries.
    #[test]
    fn smoke_test_empty() {
        let result = Update::<TestEntry>::new(vec![], 4);

        assert_eq!(result.to_evict, Vec::new());
        assert_eq!(result.to_touch, Vec::new());
    }

    /// Also no-op when the number of entries matches the capacity
    /// (desired size).
    #[test]
    fn smoke_test_at_capacity() {
        let result = Update::new(vec![TestEntry(0, true), TestEntry(1, false)], 2);

        assert_eq!(result.to_evict, Vec::new());
        assert_eq!(result.to_touch, Vec::new());
    }

    /// Evict the first entry: it hasn't been touched.
    #[test]
    fn evict_first_entry() {
        let result = Update::new(
            vec![TestEntry(0, false), TestEntry(1, true), TestEntry(2, false)],
            2,
        );

        assert_eq!(result.to_evict, vec![TestEntry(0, false)]);
        assert_eq!(result.to_touch, Vec::new());
    }

    /// Evict the first entry, despite being defined out of order.
    #[test]
    fn evict_first_entry_unsorted() {
        let result = Update::new(
            vec![TestEntry(2, false), TestEntry(1, true), TestEntry(0, false)],
            2,
        );

        assert_eq!(result.to_evict, vec![TestEntry(0, false)]);
        assert_eq!(result.to_touch, Vec::new());
    }

    /// Touch the first entry (it has been accessed), evict the second.
    #[test]
    fn evict_second_entry() {
        let result = Update::new(
            vec![TestEntry(0, true), TestEntry(1, false), TestEntry(2, false)],
            2,
        );

        assert_eq!(result.to_evict, vec![TestEntry(1, false)]);
        assert_eq!(result.to_touch, vec![TestEntry(0, true)]);
    }

    /// Evict all the unaccessed pages, find we still have to evict
    /// more, so evict from the prospective `to_touch` list.
    #[test]
    fn evict_second_pass() {
        let result = Update::new(
            vec![TestEntry(0, true), TestEntry(1, false), TestEntry(2, true)],
            1,
        );

        assert_eq!(
            result.to_evict,
            vec![TestEntry(1, false), TestEntry(0, true)]
        );
        assert_eq!(result.to_touch, vec![TestEntry(2, true)]);
    }

    /// See what happens when all the entries have been accessed.
    /// We'll want to evict the oldest, and touch the rest, in order.
    #[test]
    fn evict_all_touched() {
        let result = Update::new(
            vec![TestEntry(1, true), TestEntry(2, true), TestEntry(0, true)],
            2,
        );

        assert_eq!(result.to_evict, vec![TestEntry(0, true)]);
        assert_eq!(
            result.to_touch,
            vec![TestEntry(1, true), TestEntry(2, true)]
        );
    }

    proptest! {
        /// We can figure out the list to evict by sorting on the
        /// "accessed" `bool` flag and the `order`, and evicting the
        /// first few elements in that sorted list.
        #[test]
        fn test_eviction_oracle(mut inputs in vec(any::<TestEntry>(), 0..20usize),
                                capacity in 1..10usize) {
            let result = Update::new(inputs.clone(), capacity);

            inputs.sort_by(|x, y| (x.1, x.0).cmp(&(y.1, y.0)));

            let num_evictions = inputs.len().saturating_sub(capacity);
            assert_eq!(&result.to_evict, &inputs[0..num_evictions]);
        }

        /// Assuming the list of evictions is valid, we can figure out
        /// the entries that must be moved back to the end of the
        /// list: it's everything with the access bit set, and a
        /// timestamp less than or equal to the oldest evicted entry.
        #[test]
        fn test_touch_oracle(mut inputs in vec(any::<TestEntry>(), 0..20usize),
                             capacity in 1..10usize) {
            let result = Update::new(inputs.clone(), capacity);

            inputs.sort();

            if result.to_evict.is_empty() {
                // No eviction -> nothing to do.
                assert_eq!(result.to_touch, vec![]);
            } else if result.to_evict.iter().any(|e| e.accessed()) {
                // We evicted something with the access bit set.  We
                // must touch everything else.
                let evicted: HashSet<_> = result.to_evict.iter().cloned().collect();

                let expected: Vec<_> = inputs
                    .iter()
                    .filter(|e| !evicted.contains(e))
                    .cloned()
                    .collect();
                assert_eq!(result.to_touch, expected);
            } else {
                // All the evictions are for entries without the
                // access bit set.  Everything with the access bit set
                // and timestamp less than the most recent evictee
                // must be touched.
                let max_ts = result
                    .to_evict
                    .iter()
                    .map(|e| e.rank())
                    .max()
                    .expect("to_evict isn't empty");
                let must_touch: Vec<_> = inputs
                    .iter()
                    .filter(|e| e.accessed() && e.rank() < max_ts)
                    .cloned()
                    .collect();
                assert_eq!(&result.to_touch[0..must_touch.len()], &must_touch);

                // And entries with the timestamp *equal* to the most
                // recent evictee *may* be touched.
                let may_touch: Vec<_> = inputs
                    .iter()
                    .filter(|e| e.accessed() && e.rank() <= max_ts)
                    .cloned()
                    .collect();
                assert_eq!(&result.to_touch, &may_touch[0..result.to_touch.len()]);
            }
        }
    }
}
