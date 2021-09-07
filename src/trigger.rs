//! The probabilistic trigger lets code trigger behaviour at a goal
//! period (e.g., updating a second chance cache roughly every 10
//! writes).  It is randomised, so is not vulnerable to trivial edge
//! cases like consistently restarting a program after 9 writes and
//! never triggering a cache update.
//!
//! Sampling from an exponential would be easy to analyse, and makes
//! it clear that we can use the same state for an arbitrary number of
//! triggers, each with different periods.  However, that leaves the
//! door open for very long, if rare, gaps between triggers.
//!
//! This module instead samples from a uniform, and guarantees that
//! the probability of triggering behaviour for target period `n` is
//! never less than the `1/n` guaranteed by the exponential.
use std::cell::RefCell;

// This counter never has a zero value, except when uninitialised.
// Whenever we would decrement it to 0 (or less), we instead trigger
// the periodic behaviour and regenerate a new uniform.
std::thread_local! {
    static COUNTER: RefCell<u64> = RefCell::new(0);
}

/// Resets `c` with a new positive random uniform value and returns
/// that new value.
fn regenerate(c: &RefCell<u64>) -> u64 {
    use rand::RngCore;

    let mut rng = rand::thread_rng();

    loop {
        let rnd = rng.next_u64();
        if rnd > 0 {
            c.replace(rnd);
            return rnd;
        }
    }
}

/// Decrements the counter by `weight`.  Returns true (and resets the
/// counter) if the result would be non-positive.
fn observe(weight: u64) -> bool {
    COUNTER.with(|c| {
        let current = *c.borrow();
        if current > weight {
            c.replace(current - weight);
            return false;
        }

        let updated = regenerate(c);
        // Non-zero means `COUNTER` was initialised on entry, so we
        // can immediately return true.
        if current > 0 {
            return true;
        }

        // This is the first call to `observe` in the current thread.
        // Now that we have a random update, let's see if we trigger
        // based on the newly initialised counter.
        if updated > weight {
            // Nope.  Store the remaining weight and return false.
            c.replace(updated - weight);
            return false;
        }

        // We randomly initialised `COUNTER` and immediately
        // triggered.  Re-initialise and return `true`.
        regenerate(c);
        true
    })
}

/// A PeriodicTrigger is configured to return true roughly every
/// `period` events.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub(crate) struct PeriodicTrigger {
    // `scale * period = u64:MAX` (roughly).
    scale: u64,
}

impl PeriodicTrigger {
    /// Returns a `PeriodicTrigger` that will trigger roughly every
    /// `period` events.
    pub fn new(mut period: u64) -> PeriodicTrigger {
        // It doesn't make sense to trigger every 0 event; let's say
        // the caller meant to trigger every event (i.e., always).
        if period == 0 {
            period = 1;
        }

        let scale = (u64::MAX / period) + ((u64::MAX % period) > 0) as u64;
        PeriodicTrigger { scale }
    }

    /// Observes one "event".  Returns whether the periodic trigger
    /// fired for this event (i.e., whether the caller should invoke
    /// the periodic behaviour).
    #[inline(always)]
    pub fn event(self) -> bool {
        self.weighted_event(1)
    }

    /// Observes count "events".  Returns whether the periodic trigger
    /// fired for any of these events (i.e., whether the caller should
    /// invoke the periodic behaviour).
    #[inline(always)]
    pub fn weighted_event(self, count: u64) -> bool {
        observe(self.scale.saturating_mul(count))
    }
}

#[test]
fn smoke_test() {
    let trigger = PeriodicTrigger::new(10);

    // We should never see a single failure.
    for _ in 0..10 {
        assert!((0..10).any(|_| trigger.event()))
    }
}

#[test]
fn test_delay_until_trigger() {
    let trigger = PeriodicTrigger::new(10);

    // Find the max delay before we fire.
    let mut max_delay = 0;
    for _ in 0..200 {
        let mut triggered = false;
        for i in 1..=10 {
            triggered = trigger.event();
            if triggered {
                max_delay = max_delay.max(i);
                break;
            }
        }

        // We never need more than 10 events.
        assert!(triggered);
    }

    // We should see a trigger after 9 or 10 calls with 20%
    // probability.  After 200 attempts, we'll find at least
    // one such call with more than 19 nines.
    assert!(max_delay >= 9);
}

#[test]
fn test_weighted_delay_until_trigger() {
    let trigger = PeriodicTrigger::new(10);

    // Find the max delay before we fire.
    let mut max_delay = 0;
    for _ in 0..200 {
        let mut triggered = false;
        for i in 1..=5 {
            triggered = trigger.weighted_event(2);
            if triggered {
                max_delay = max_delay.max(i);
                break;
            }
        }

        // We never need more than 5 events.
        assert!(triggered);
    }

    assert!(max_delay >= 4);
}

#[test]
fn test_weighted_always_fire() {
    let trigger = PeriodicTrigger::new(20);

    for _ in 0..10 {
        assert!(trigger.weighted_event(20));
    }
}

#[test]
fn test_zero_period() {
    let trigger = PeriodicTrigger::new(0);

    // The first event should always trigger.
    for _ in 0..10 {
        assert!(trigger.event());
    }
}

#[test]
fn test_one_period() {
    let trigger = PeriodicTrigger::new(1);

    // The first event should always trigger.
    for _ in 0..10 {
        assert!(trigger.event());
    }
}

#[test]
fn test_infinity_period() {
    let trigger = PeriodicTrigger::new(u64::MAX);

    // It should virtually never trigger.
    for _ in 0..1000 {
        assert!(!trigger.event());
    }
}
