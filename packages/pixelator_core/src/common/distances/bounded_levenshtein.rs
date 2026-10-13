//! Single-word Myers bit-parallel bounded infix Levenshtein search.
//!
//! The pattern and text are both 2-bit-packed DNA (≤ 32 bases each, one `u64`
//! word; same little-endian packing as [`crate::encoding::encode_dna_2bit`]).
//! The search reports whether the *whole* pattern aligns to *any substring* of
//! the text within a bounded number of edits (substitutions + indels), i.e. an
//! infix (text-local) Levenshtein match with free end-gaps on the text side only.
//!
//! # Algorithm
//!
//! This is Myers' 1999 infix (semi-global) approximate string matching: after
//! column `j`, `score` is the edit distance from the whole pattern to the best
//! text substring *ending* at `j`. We short-circuit as soon as
//! `score <= max_edits`. The horizontal shift is a plain `<< 1` with no `| 1`,
//! so row 0 stays 0 (report-all-occurrences / infix form).
//!
//! The bit-vector recurrences follow Hyyrö's D0 reformulation of Myers (same
//! algorithm, different spelling). This is **not** Wu–Manber/`agrep` NFA
//! matching, **not** global end-to-end Levenshtein, and **not** Damerau
//! (no transpositions).
//!
//! # References
//!
//! - Myers, G. (1999). A fast bit-vector algorithm for approximate string
//!   matching based on dynamic programming. *J. ACM* 46(3):395–415.
//!   <https://doi.org/10.1145/316542.316550>
//! - Hyyrö, H. (2001). Explaining and Extending the Bit-Parallel Approximate
//!   String Matching Algorithm of Myers. University of Tampere, Dept. of
//!   Computer and Information Sciences, Report A-2001-10.
//!   <https://researchportal.tuni.fi/en/publications/explaining-and-extending-the-bit-parallel-approximate-string-matc/>
//!
//! # Name map (ours → Myers 1999 → Hyyrö 2001)
//!
//! - `peq` → Peq → Peq
//! - `vp` → Pv → VP
//! - `vn` → Mv → VN
//! - `eq` → Eq → (pattern-match bits for the current text base)
//! - `x` → Xv → Xv
//! - `d0` → (related to Xh; D0 ≈ Xv | Xh) → D0
//! - `hp` → Ph → HP
//! - `hn` → Mh → HN
//! - `score` → Score → D[m, j]

/// Returns whether the packed `pattern` (low `2 * pattern_len` bits, base 0 in
/// the lowest bits) aligns to any substring of the packed `text` (low
/// `2 * text_len` bits) within `max_edits` edits (infix Levenshtein;
/// substitutions + indels).
///
/// Both `pattern_len` and `text_len` must be ≤ 32 (one `u64` of 2-bit codes).
///
/// Instead of filling an `(m+1)×(n+1)` DP table, each column's vertical delta
/// bits are packed into a `u64` and advanced one text base at a time with
/// word-wide bit ops — O(n) for patterns of length ≤ 32. See the module docs
/// for the Myers / Hyyrö citations and name map.
pub fn within_bounded_levenshtein(
    pattern: u64,
    pattern_len: usize,
    text: u64,
    text_len: usize,
    max_edits: u32,
) -> bool {
    debug_assert!(pattern_len <= 32 && text_len <= 32);

    let m = pattern_len;
    // Empty pattern aligns to the empty text substring with zero edits.
    if m == 0 {
        return true;
    }

    // `peq[c]` has bit `i` set iff pattern base `i` has 2-bit code `c`.
    let mut peq = [0u64; 4];
    for i in 0..m {
        let c = ((pattern >> (2 * i)) & 0b11) as usize;
        peq[c] |= 1u64 << i;
    }

    // All vectors are kept to `m` bits; masking after every `!`/`<<` keeps the
    // high garbage out of the `(Eq & VP) + VP` carry chain, which otherwise
    // wraps around the word and corrupts the low bits.
    let mask: u64 = if m >= 64 { u64::MAX } else { (1u64 << m) - 1 };
    let top_bit = 1u64 << (m - 1);
    let mut vp: u64 = mask;
    let mut vn: u64 = 0;
    // Score against the empty text prefix: every pattern base is a deletion.
    // I.e. the score is the number of deletions needed to align the pattern to the empty text.
    let mut score = m as u32;
    if score <= max_edits {
        return true;
    }

    let n = text_len;
    for j in 0..n {
        // Canonical Myers "report all occurrences" (infix) step: the row-0
        // boundary stays 0, so the horizontal shift is a plain `<< 1` with no
        // `| 1` (that extra bit is only for whole-string edit distance).
        // Hyyrö D0 form: x = eq | vn; d0 = ((vp + (x & vp)) ^ vp) | x.
        let code = ((text >> (2 * j)) & 0b11) as usize;
        let eq = peq[code];
        let x = eq | vn;
        let d0 = ((vp.wrapping_add(x & vp)) ^ vp) | x;
        let hn = vp & d0;
        let hp = vn | !(d0 | vp);

        if hp & top_bit != 0 {
            score += 1;
        } else if hn & top_bit != 0 {
            score -= 1;
        }

        if score <= max_edits {
            return true;
        }

        // The bottom-row score moves by at most one per column, so if even
        // driving it down every remaining column cannot reach the budget,
        // there is nothing left to find.
        let remaining = (n - j - 1) as u32;
        if score - max_edits > remaining {
            return false;
        }

        let hp_shifted = (hp << 1) & mask;
        vn = hp_shifted & d0;
        vp = ((hn << 1) | !(hp_shifted | d0)) & mask;
    }

    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::encode_dna_2bit;

    fn matches(pattern: &[u8], text: &[u8], k: u32) -> bool {
        within_bounded_levenshtein(
            encode_dna_2bit(pattern),
            pattern.len(),
            encode_dna_2bit(text),
            text.len(),
            k,
        )
    }

    #[test]
    fn exact_substring_matches_at_zero_edits() {
        assert!(matches(b"ACGTACGT", b"ACGTACGT", 0));
        assert!(matches(b"ACGTACGT", b"TTTTACGTACGTTTT", 0));
    }

    #[test]
    fn single_substitution_needs_one_edit() {
        // Best substring "ACGTACCT" is one substitution from the pattern.
        assert!(!matches(b"ACGTACGT", b"TTTTACGTACCTTTT", 0));
        assert!(matches(b"ACGTACGT", b"TTTTACGTACCTTTT", 1));
    }

    #[test]
    fn single_deletion_needs_one_edit() {
        // Text "ACT" is the pattern "ACGT" with the G deleted.
        assert!(!matches(b"ACGT", b"ACT", 0));
        assert!(matches(b"ACGT", b"ACT", 1));
    }

    #[test]
    fn single_insertion_needs_one_edit() {
        // Text substring "ACAGT" is the pattern "ACGT" with an extra A inserted;
        // "ACGT" itself is not a contiguous substring here.
        assert!(!matches(b"ACGT", b"GGACAGTGG", 0));
        assert!(matches(b"ACGT", b"GGACAGTGG", 1));
    }

    #[test]
    fn distant_pattern_rejected_under_small_budget() {
        // "ACGTACGT" vs an all-G text: G matches only at pattern indices 2 and 6,
        // so at least six substitutions are required.
        assert!(!matches(b"ACGTACGT", b"GGGGGGGG", 1));
        assert!(!matches(b"ACGTACGT", b"GGGGGGGG", 5));
        assert!(matches(b"ACGTACGT", b"GGGGGGGG", 6));
    }

    #[test]
    fn empty_text_needs_full_length_deletions() {
        // With no text, the only alignment deletes every pattern base.
        assert!(!matches(b"ACGT", b"", 3));
        assert!(matches(b"ACGT", b"", 4));
    }

    #[test]
    fn empty_pattern_always_matches() {
        // Empty pattern aligns to the empty substring with zero edits.
        assert!(matches(b"", b"", 0));
        assert!(matches(b"", b"ACGT", 0));
    }

    #[test]
    fn budget_at_least_pattern_length_matches_without_scan() {
        // Score starts at m; if max_edits >= m the empty alignment already wins.
        // Homopolymer vs its complement has infix distance m, so k = m-1 fails.
        assert!(!matches(b"AAAA", b"CCCC", 3));
        assert!(matches(b"AAAA", b"CCCC", 4));
    }

    #[test]
    fn exact_match_at_text_start_and_end() {
        assert!(matches(b"ACGT", b"ACGTTTTT", 0));
        assert!(matches(b"ACGT", b"TTTTACGT", 0));
        // Near-miss only at the opposite end still fails at zero edits.
        assert!(!matches(b"ACGT", b"ACGCTTTT", 0));
        assert!(!matches(b"ACGT", b"TTTTACGC", 0));
    }

    #[test]
    fn two_substitutions_need_two_edits() {
        // Best substring "ACCTACCT" differs from "ACGTACGT" at two sites.
        assert!(!matches(b"ACGTACGT", b"TTTTACCTACCTTTTT", 1));
        assert!(matches(b"ACGTACGT", b"TTTTACCTACCTTTTT", 2));
    }

    #[test]
    fn substitution_plus_deletion_needs_two_edits() {
        // "ATT" from "ACGT": delete C and substitute G→T (length gap blocks any
        // single-edit path).
        assert!(!matches(b"ACGT", b"GGATTGG", 1));
        assert!(matches(b"ACGT", b"GGATTGG", 2));
    }

    #[test]
    fn length_32_exact_and_near_miss() {
        let pattern = b"ACGTACGTACGTACGTACGTACGTACGTACGT";
        assert_eq!(pattern.len(), 32);
        assert!(matches(pattern, pattern, 0));

        // One substitution in the middle of an otherwise exact text.
        let mut near = pattern.to_vec();
        near[16] = b'T'; // was A
        assert!(!matches(pattern, &near, 0));
        assert!(matches(pattern, &near, 1));

        // Unrelated length-32 text needs many edits; reject under a tight budget.
        let distant = b"GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG";
        assert!(!matches(pattern, distant, 1));
    }

    #[test]
    fn repeated_bases_exercise_multi_bit_peq() {
        // Homopolymer pattern: peq[A] has several bits set.
        assert!(matches(b"AAAA", b"AAAA", 0));
        // One deletion short of the pattern.
        assert!(!matches(b"AAAA", b"AAA", 0));
        assert!(matches(b"AAAA", b"AAA", 1));
        // One insertion; exact "AAAA" is still a substring of "AAAAA".
        assert!(matches(b"AAAA", b"AAAAA", 0));
    }

    #[test]
    fn infix_match_ignores_flanking_text() {
        // Pattern is an interior exact substring; free end-gaps on the text side.
        assert!(matches(b"ACGT", b"TTTTACGTAAAA", 0));
        // Same pattern is not a prefix or suffix of the full text.
        assert!(!b"TTTTACGTAAAA".starts_with(b"ACGT"));
        assert!(!b"TTTTACGTAAAA".ends_with(b"ACGT"));
    }
}
