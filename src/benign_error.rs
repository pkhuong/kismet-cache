/// We expect some [`std::io::Error`] during regular operations:
/// Kismet relies on the filesystem for concurrency control.
use std::io::Error;
use std::io::ErrorKind;

/// Checks whether the error is for a missing file: NotFound, or stale
/// handle.  A stale (NFS) handle means the inode we're trying to read
/// isn't available on the server anymore.  Maybe we'd find something
/// else if we flushed our client's filehandle cache, but things do go
/// missing from caches, so gracefully treating stale handles like
/// cache misses should be fine.
pub fn is_absent_file_error(error: &Error) -> bool {
    if error.kind() == ErrorKind::NotFound {
        true
    } else if let Some(errno) = error.raw_os_error() {
        // We'd like to use [`ErrorKind::StaleNetworkFileHandle`],
        // but that's not stabilised https://github.com/rust-lang/rust/issues/86442
        errno == libc::ESTALE
    } else {
        false
    }
}

// Mostly trivial, but let's at least make sure we didn't mess up raw_os_error
// and confirm that libc agrees with what we know to be true on Linux.
#[test]
fn test_getters() {
    assert_eq!(
        true,
        is_absent_file_error(&Error::new(ErrorKind::NotFound, "not found"))
    );
    assert_eq!(
        false,
        is_absent_file_error(&Error::new(ErrorKind::PermissionDenied, "bad"))
    );

    assert_eq!(
        true,
        is_absent_file_error(&Error::from_raw_os_error(libc::ENOENT))
    );
    assert_eq!(
        true,
        is_absent_file_error(&Error::from_raw_os_error(libc::ESTALE))
    );
    assert_eq!(
        false,
        is_absent_file_error(&Error::from_raw_os_error(libc::EIO))
    );

    #[cfg(target_os = "linux")]
    assert_eq!(true, is_absent_file_error(&Error::from_raw_os_error(2))); // ENOENT
    #[cfg(target_os = "linux")]
    assert_eq!(true, is_absent_file_error(&Error::from_raw_os_error(116))); // ESTALE
    #[cfg(target_os = "linux")]
    assert_eq!(false, is_absent_file_error(&Error::from_raw_os_error(1))); // EPERM
}
