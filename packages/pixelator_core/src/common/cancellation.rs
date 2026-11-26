//! Cooperative cancellation of long-running algorithms.
//!
//! A caller wraps a computation in [`run_with`], handing it a [`CancellationToken`]. Hot loops
//! inside the computation call [`checkpoint`]; once the token is cancelled, the next checkpoint
//! unwinds the stack and [`run_with`] returns `Err(Cancelled)`. The algorithms themselves need
//! neither a token parameter nor a `Result` return type.
//!
//! This relies on unwinding, so the crate cannot be built with `panic = "abort"`.
//!
//! The token is stored per thread: checkpoints reached from other threads (e.g. rayon workers)
//! are no-ops.

use std::cell::RefCell;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

#[cfg(not(panic = "unwind"))]
compile_error!(
    "cancellation relies on unwinding; do not build this crate with `panic = \"abort\"`"
);

/// A flag that [`checkpoint`] polls to find out whether the running computation should stop.
///
/// Clones share the same flag, so one clone can be handed to [`run_with`] while another is
/// kept to call [`cancel`](Self::cancel) from a different thread.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken(Arc<AtomicBool>);

impl CancellationToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

/// Returned by [`run_with`] when the computation stopped because its token was cancelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Cancelled;

impl fmt::Display for Cancelled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "the computation was cancelled")
    }
}

impl std::error::Error for Cancelled {}

thread_local! {
    static CURRENT: RefCell<Option<CancellationToken>> = const { RefCell::new(None) };
}

/// Restores the previously installed token when dropped, including during unwinding.
struct Scope(Option<CancellationToken>);

impl Scope {
    fn enter(token: CancellationToken) -> Self {
        Self(CURRENT.with(|current| current.replace(Some(token))))
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        CURRENT.with(|current| *current.borrow_mut() = self.0.take());
    }
}

/// Runs `f` with `token` installed for the current thread.
///
/// Returns `Err(Cancelled)` if a [`checkpoint`] reached from `f` saw the token cancelled. Any
/// other panic is propagated unchanged. State that `f` was mutating when it was cancelled may
/// be half-updated and should be discarded.
pub fn run_with<T>(token: CancellationToken, f: impl FnOnce() -> T) -> Result<T, Cancelled> {
    let _scope = Scope::enter(token);
    catch_unwind(AssertUnwindSafe(f)).map_err(|payload| match payload.downcast::<Cancelled>() {
        Ok(_) => Cancelled,
        Err(other) => resume_unwind(other),
    })
}

/// Stops the computation started by [`run_with`] if its token has been cancelled.
///
/// Does nothing when no token is installed on the current thread. Unwinds with
/// [`resume_unwind`] rather than `panic!` so that the panic hook does not print a message.
#[inline]
pub fn checkpoint() {
    let cancelled = CURRENT.with(|current| {
        current
            .borrow()
            .as_ref()
            .is_some_and(CancellationToken::is_cancelled)
    });
    if cancelled {
        resume_unwind(Box::new(Cancelled));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_token_is_not_cancelled() {
        assert!(!CancellationToken::new().is_cancelled());
    }

    #[test]
    fn test_cancel_is_seen_by_clones() {
        let token = CancellationToken::new();
        let clone = token.clone();
        clone.cancel();
        assert!(token.is_cancelled());
    }

    #[test]
    fn test_cancel_is_seen_across_threads() {
        let token = CancellationToken::new();
        let clone = token.clone();
        std::thread::spawn(move || clone.cancel()).join().unwrap();
        assert!(token.is_cancelled());
    }

    #[test]
    fn test_checkpoint_without_token_is_a_noop() {
        checkpoint();
    }

    #[test]
    fn test_run_with_uncancelled_token_returns_value() {
        assert_eq!(run_with(CancellationToken::new(), || 42), Ok(42));
    }

    #[test]
    fn test_run_with_cancelled_token_stops_at_checkpoint() {
        let token = CancellationToken::new();
        token.cancel();
        let mut reached_end = false;

        let result = run_with(token, || {
            checkpoint();
            reached_end = true;
        });

        assert_eq!(result, Err(Cancelled));
        assert!(!reached_end);
    }

    #[test]
    fn test_run_with_stops_when_cancelled_mid_run() {
        let token = CancellationToken::new();
        let canceller = token.clone();
        let mut iterations = 0;

        let result = run_with(token, || {
            loop {
                iterations += 1;
                if iterations == 3 {
                    canceller.cancel();
                }
                checkpoint();
            }
        });

        assert_eq!(result, Err(Cancelled));
        assert_eq!(iterations, 3);
    }

    #[test]
    fn test_run_with_drops_locals_when_cancelled() {
        struct SetOnDrop<'a>(&'a AtomicBool);
        impl Drop for SetOnDrop<'_> {
            fn drop(&mut self) {
                self.0.store(true, Ordering::Relaxed);
            }
        }
        let dropped = AtomicBool::new(false);
        let token = CancellationToken::new();
        token.cancel();

        let _ = run_with(token, || {
            let _guard = SetOnDrop(&dropped);
            checkpoint();
        });

        assert!(dropped.load(Ordering::Relaxed));
    }

    #[test]
    fn test_run_with_propagates_other_panics() {
        let result = catch_unwind(|| run_with(CancellationToken::new(), || panic!("a real bug")));
        let payload = result.unwrap_err();
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"a real bug"));
    }

    #[test]
    fn test_token_is_uninstalled_after_run_with() {
        let token = CancellationToken::new();
        token.cancel();
        let _ = run_with(token, checkpoint);

        // A cancelled token must not leak into later work on the same thread.
        checkpoint();
    }

    #[test]
    fn test_nested_run_with_restores_outer_token() {
        let outer = CancellationToken::new();
        let result = run_with(outer.clone(), || {
            let inner = CancellationToken::new();
            inner.cancel();
            assert_eq!(run_with(inner, checkpoint), Err(Cancelled));

            outer.cancel();
            checkpoint();
        });
        assert_eq!(result, Err(Cancelled));
    }

    #[test]
    fn test_checkpoint_is_a_noop_on_other_threads() {
        let token = CancellationToken::new();
        token.cancel();
        let result = run_with(token, || std::thread::spawn(checkpoint).join().is_ok());
        assert_eq!(result, Ok(true));
    }
}
