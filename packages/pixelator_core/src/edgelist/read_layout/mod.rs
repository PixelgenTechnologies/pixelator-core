//! How a raw read is laid out: the vocabulary for locating a region within it.
//!
//! A region's position depends on two independent axes - which conjugate it
//! belongs to ([`ConjugateSide`]) and which orientation the read was sequenced
//! in ([`Strand`]) - and [`RegionLayout`] resolves that pair into byte offsets.
//! [`Read`] is the unit those offsets apply to.
//!
//! These types are shared vocabulary rather than the property of any one stage:
//! extraction and quality profiling both address reads this way, so neither owns
//! them. Nothing here depends on the panel or on extraction, which keeps the
//! dependency arrows pointing inward from both consumers.

mod conjugate;
mod layout;
mod read;
mod strand;

pub use conjugate::ConjugateSide;
pub use read::Read;
pub use strand::Strand;

// Narrower than its neighbours: the resolved offsets are an internal detail
// shared between pipeline stages, not part of the crate's surface.
pub(crate) use layout::RegionLayout;
