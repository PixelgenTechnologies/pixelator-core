/// Hamming distance between two 2-bit-packed DNA sequences: the number of
/// differing symbols (substitutions only; not Levenshtein).
pub fn hamming_distance_2bit(a: u64, b: u64) -> u8 {
    // Count differing 2-bit symbols rather than differing bits.
    let mut x = a ^ b;
    x |= x >> 1;
    (x & 0x5555_5555_5555_5555).count_ones() as u8
}

/// Returns whether two 2-bit-packed DNA sequences differ in at most
/// `max_distance` symbols (Hamming / substitutions only; not Levenshtein).
pub fn within_hamming_distance(a: u64, b: u64, max_distance: u8) -> bool {
    // Compute the mismatch mask at 2-bit-symbol granularity.
    let mut mismatches = a ^ b;
    mismatches |= mismatches >> 1;
    mismatches &= 0x5555_5555_5555_5555;

    // Early exit by clearing one mismatch bit at a time.
    // If bits remain after max_distance removals, distance is larger.
    for _ in 0..max_distance {
        if mismatches == 0 {
            return true;
        }
        mismatches &= mismatches - 1;
    }

    mismatches == 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoding::encode_dna_2bit;

    #[test]
    fn hamming_distance_counts_symbol_mismatches() {
        let a = encode_dna_2bit(b"ACGTAC");
        let b = encode_dna_2bit(b"ACATAG");
        let c = encode_dna_2bit(b"TCATAG");

        assert_eq!(hamming_distance_2bit(a, b), 2);
        assert_eq!(hamming_distance_2bit(a, c), 3);
    }

    #[test]
    fn hamming_distance_is_symmetric_and_zero_for_equal_values() {
        let a = encode_dna_2bit(b"GATTACA");
        let b = encode_dna_2bit(b"GACTATA");

        assert_eq!(hamming_distance_2bit(a, a), 0);
        assert_eq!(hamming_distance_2bit(a, b), hamming_distance_2bit(b, a));
    }

    #[test]
    fn hamming_distance_counts_all_mismatches_for_same_length_sequences() {
        let a = encode_dna_2bit(b"AAAAAA");
        let b = encode_dna_2bit(b"TTTTTT");
        assert_eq!(hamming_distance_2bit(a, b), 6);
    }

    #[test]
    fn within_hamming_distance_respects_threshold_boundaries() {
        let a = encode_dna_2bit(b"ACGTAC");
        let b = encode_dna_2bit(b"ACATAG"); // distance 2
        let c = encode_dna_2bit(b"TCATAG"); // distance 3

        assert!(within_hamming_distance(a, b, 2));
        assert!(within_hamming_distance(a, b, 3));
        assert!(!within_hamming_distance(a, c, 2));
        assert!(within_hamming_distance(a, c, 3));
    }

    #[test]
    fn within_hamming_distance_matches_exact_distance_for_varied_inputs() {
        let pairs: [(&[u8], &[u8]); 5] = [
            (b"AAAAAAAA", b"AAAAAAAA"),
            (b"AAAAAAAA", b"AAAAAAAT"),
            (b"AAAAAAAA", b"AAAATTTT"),
            (b"ACGTACGT", b"TGCATGCA"),
            (b"GATTACAG", b"GACTATAG"),
        ];

        for (lhs, rhs) in pairs {
            let a = encode_dna_2bit(lhs);
            let b = encode_dna_2bit(rhs);
            let distance = hamming_distance_2bit(a, b);

            for threshold in 0..=8 {
                assert_eq!(
                    within_hamming_distance(a, b, threshold),
                    distance <= threshold
                );
            }
        }
    }
}
