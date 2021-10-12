Kismet: a Second Chance cache for files on disk
===============================================
[![Build Status](https://app.travis-ci.com/pkhuong/kismet-cache.svg?branch=main)](https://app.travis-ci.com/pkhuong/kismet-cache) [![Coverage Status](https://coveralls.io/repos/github/pkhuong/kismet-cache/badge.svg?branch=main)](https://coveralls.io/github/pkhuong/kismet-cache?branch=main)

Kismet implements multiprocess lock-free[^lock-free-fs] crash-safe and
(roughly) bounded persistent caches stored in filesystem directories,
with a [Second Chance](https://en.wikipedia.org/wiki/Page_replacement_algorithm#Second-chance)
eviction strategy.  The maintenance logic is batched and invoked at
periodic jittered intervals to make sure accesses amortise to a
constant number of filesystem system calls and logarithmic (in the
number of cached file) time complexity.  That's good for performance,
and enables lock-freedom,[^unlike-ccache] but does mean that caches
are expected to temporarily grow past their capacity hlimits, although
rarely by more than a factor of 2 or 3.
