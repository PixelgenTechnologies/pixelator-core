use paraseq::Record;

/// The raw input for one molecule: either a single read spanning the whole
/// amplicon, or a pair sequenced from both ends.
///
/// Single-end variants carry the orientation they were sequenced in, so a
/// consumer knows which [`Strand`](super::Strand) bounds to address them with;
/// paired reads always run forward in `r1` and reverse in `r2`.
pub enum Read<'a, R: Record> {
    SingleForward(&'a R),
    SingleReverse(&'a R),
    Paired { r1: &'a R, r2: &'a R },
}
