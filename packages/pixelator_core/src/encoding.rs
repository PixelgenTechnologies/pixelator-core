/// Maps a single DNA base (as an ASCII byte) to its 2-bit code, panicking on
/// non-ACGT input.
///
/// Matches the Python LUT from pixelator:
/// A -> [0, 0], C -> [1, 0], G -> [0, 1], T -> [1, 1]
fn base_to_2bit(base: u8) -> u64 {
    match base {
        b'A' => 0b00u64,
        b'C' => 0b01u64,
        b'G' => 0b10u64,
        b'T' => 0b11u64,
        _ => panic!("Unsupported DNA base: {}", base as char),
    }
}

/// Packs a DNA sequence into a 2-bit-per-base integer in little-endian symbol
/// order (the first base occupies the lowest bits). Panics on non-ACGT input.
///
/// The result is a `u64`, so sequences of up to 32 bases fit; longer sequences
/// overflow the shift.
pub fn encode_dna_2bit(dna: &[u8]) -> u64 {
    dna.iter().enumerate().fold(0u64, |acc, (idx, &base)| {
        acc | (base_to_2bit(base) << (2 * idx))
    })
}

/// Encodes the reverse complement of a DNA sequence directly, without building
/// an intermediate buffer. Panics on non-ACGT input.
///
/// In this scheme, complementing a base is a XOR with `0b11` (A<->T, C<->G),
/// and reversing places the base at index `idx` at symbol position
/// `n - 1 - idx`.
pub fn encode_dna_2bit_rc(dna: &[u8]) -> u64 {
    let n = dna.len();
    dna.iter().enumerate().fold(0u64, |acc, (idx, &base)| {
        acc | ((base_to_2bit(base) ^ 0b11) << (2 * (n - 1 - idx)))
    })
}

/// Inverse of [`base_to_2bit`]: maps a 2-bit code back to an ASCII DNA base.
fn bit_to_base(bits: u8) -> u8 {
    match bits {
        0b00 => b'A',
        0b01 => b'C',
        0b10 => b'G',
        0b11 => b'T',
        _ => unreachable!("2-bit mask yields at most 0b11"),
    }
}

/// Unpacks a 2-bit-per-base integer back into `length` DNA bases, the inverse of
/// [`encode_dna_2bit`]. Bits above `2 * length` are ignored.
///
/// Panics if `length` exceeds 32, the most bases a `u64` can hold.
pub fn decode_dna_2bit(encoded: u64, length: usize) -> Vec<u8> {
    assert!(length <= 32, "Cannot decode {length} bases from a u64");
    (0..length)
        .map(|idx| bit_to_base(((encoded >> (2 * idx)) & 0b11) as u8))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_dna_2bit_matches_base_mapping() {
        assert_eq!(encode_dna_2bit(b"A"), 0);
        assert_eq!(encode_dna_2bit(b"C"), 1);
        assert_eq!(encode_dna_2bit(b"G"), 2);
        assert_eq!(encode_dna_2bit(b"T"), 3);
    }

    #[test]
    fn encode_dna_2bit_packs_bases_in_little_endian_symbol_order() {
        // "ACGT" -> A(00), C(01), G(10), T(11)
        // packed as: 00 | 01<<2 | 10<<4 | 11<<6 = 0b1110_0100 = 228
        assert_eq!(encode_dna_2bit(b"ACGT"), 228);
        // Empty input should encode to 0.
        assert_eq!(encode_dna_2bit(b""), 0);
    }

    #[test]
    #[should_panic(expected = "Unsupported DNA base")]
    fn encode_dna_2bit_rejects_invalid_bases() {
        let _ = encode_dna_2bit(b"AN");
    }

    #[test]
    fn encode_dna_2bit_rc_matches_encoding_of_reverse_complement() {
        // RC("ACTTCCTAGG") == "CCTAGGAAGT".
        assert_eq!(
            encode_dna_2bit_rc(b"ACTTCCTAGG"),
            encode_dna_2bit(b"CCTAGGAAGT")
        );
        // RC of a single base is its complement.
        assert_eq!(encode_dna_2bit_rc(b"A"), encode_dna_2bit(b"T"));
        // Empty input encodes to 0.
        assert_eq!(encode_dna_2bit_rc(b""), 0);
    }

    #[test]
    #[should_panic(expected = "Unsupported DNA base")]
    fn encode_dna_2bit_rc_rejects_invalid_bases() {
        let _ = encode_dna_2bit_rc(b"AN");
    }

    #[test]
    fn decode_dna_2bit_matches_base_mapping() {
        assert_eq!(decode_dna_2bit(0, 1), b"A");
        assert_eq!(decode_dna_2bit(1, 1), b"C");
        assert_eq!(decode_dna_2bit(2, 1), b"G");
        assert_eq!(decode_dna_2bit(3, 1), b"T");
    }

    #[test]
    fn decode_dna_2bit_unpacks_little_endian_symbol_order() {
        // 228 = 0b1110_0100 -> A(00), C(01), G(10), T(11)
        assert_eq!(decode_dna_2bit(228, 4), b"ACGT");
        assert_eq!(decode_dna_2bit(0, 0), b"");
    }

    #[test]
    fn decode_dna_2bit_ignores_bits_above_length() {
        // Same packed value as "ACGT", but only the first two symbols are kept.
        assert_eq!(decode_dna_2bit(228, 2), b"AC");
    }

    #[test]
    #[should_panic(expected = "Cannot decode 33 bases from a u64")]
    fn decode_dna_2bit_rejects_lengths_beyond_u64() {
        let _ = decode_dna_2bit(0, 33);
    }

    #[test]
    fn decode_encode_round_trips() {
        let sequences: [&[u8]; 6] = [
            b"",
            b"A",
            b"ACGT",
            b"AAAAAAAA",
            b"TTTTTTTT",
            b"ACGTACGTACGTACGTACGTACGTACGTACGT", // 32 bases
        ];
        for seq in sequences {
            assert_eq!(decode_dna_2bit(encode_dna_2bit(seq), seq.len()), seq);
        }
    }

    #[test]
    fn encode_decode_round_trips_for_values() {
        let cases: [(u64, usize); 4] = [(0, 0), (0b11, 1), (228, 4), (0x0123_4567_89AB_CDEF, 32)];
        for (value, length) in cases {
            let mask = if length == 0 {
                0
            } else {
                u64::MAX >> (64 - 2 * length)
            };
            let masked = value & mask;
            assert_eq!(encode_dna_2bit(&decode_dna_2bit(masked, length)), masked);
        }
    }

    #[test]
    fn decode_of_rc_encoding_is_reverse_complement() {
        assert_eq!(
            decode_dna_2bit(encode_dna_2bit_rc(b"ACTTCCTAGG"), 10),
            b"CCTAGGAAGT"
        );
    }
}
