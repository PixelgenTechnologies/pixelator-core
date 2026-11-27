use std::io::{self, Read};

/// Longest FASTQ line accepted from a decoded input, in bytes (newline excluded).
///
/// The FASTQ parser buffers a record until it has seen all four of its lines,
/// and a worker holds batches of 1024 records, so the memory a read may take is
/// set by this ceiling. 16 KiB is far above the read and header lengths of
/// short-read sequencing, while a batch stays within 64 MiB per mate whatever
/// the input, on the order of the per-thread budget of the batch queues.
pub(crate) const MAX_FASTQ_LINE_BYTES: usize = 16 * 1024;

/// Fails the read once any line of `inner` grows past `max_line` bytes.
///
/// Wraps the decoded stream (after gzip), so a record split across gzip members
/// is measured as one, and no read takes more than the bytes left on the current
/// line: an oversized line is rejected before more than `max_line + 1` of its
/// bytes are handed to the parser.
pub(crate) struct LineLengthLimit<R> {
    inner: R,
    max_line: usize,
    /// Bytes of the current, unterminated line seen so far.
    line_len: usize,
}

impl<R> LineLengthLimit<R> {
    pub(crate) fn new(inner: R, max_line: usize) -> Self {
        Self {
            inner,
            max_line,
            line_len: 0,
        }
    }
}

impl<R: Read> Read for LineLengthLimit<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        // Once over the ceiling the stream stays failed: later reads (another
        // worker may retry) must not resume mid-line.
        if self.line_len > self.max_line {
            return Err(self.too_long());
        }
        let allowance = self.max_line - self.line_len + 1;
        let len = buf.len().min(allowance);
        let n = self.inner.read(&mut buf[..len])?;
        let chunk = &buf[..n];
        // Every line that ends inside `chunk` fits by construction of
        // `allowance`; only the unterminated tail can exceed the ceiling.
        self.line_len = match chunk.iter().rposition(|&b| b == b'\n') {
            Some(last_newline) => n - last_newline - 1,
            None => self.line_len + n,
        };
        if self.line_len > self.max_line {
            return Err(self.too_long());
        }
        Ok(n)
    }
}

impl<R> LineLengthLimit<R> {
    fn too_long(&self) -> io::Error {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "FASTQ line longer than {} bytes; the input is not short-read FASTQ",
                self.max_line
            ),
        )
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::*;

    fn read_all(input: &[u8], max_line: usize) -> io::Result<Vec<u8>> {
        let mut out = Vec::new();
        LineLengthLimit::new(Cursor::new(input.to_vec()), max_line).read_to_end(&mut out)?;
        Ok(out)
    }

    #[test]
    fn lines_up_to_the_ceiling_pass_through_unchanged() {
        let input = b"@r\nACGT\n+\nIIII\n@r2\nACG\n+\nIII";
        assert_eq!(read_all(input, 4).expect("within ceiling"), input);
    }

    #[test]
    fn a_line_past_the_ceiling_is_rejected() {
        let err = read_all(b"@r\nACGTA\n+\nIIIII\n", 4).expect_err("line of 5 > 4");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn an_unterminated_final_line_past_the_ceiling_is_rejected() {
        assert!(read_all(b"@r\nACGTA", 4).is_err());
    }

    #[test]
    fn never_hands_out_more_than_the_rest_of_an_oversized_line() {
        let mut reader = LineLengthLimit::new(Cursor::new(vec![b'A'; 1 << 20]), 8);
        let mut buf = vec![0; 1 << 20];
        assert!(reader.read(&mut buf).is_err());
        assert_eq!(
            reader.inner.position(),
            9,
            "read past the ceiling's one byte"
        );
        assert!(reader.read(&mut buf).is_err(), "the rejection is sticky");
        assert_eq!(reader.inner.position(), 9);
    }

    #[test]
    fn line_length_carries_across_reads() {
        let mut reader = LineLengthLimit::new(Cursor::new(b"AAAA\nAAAAA".to_vec()), 4);
        let mut buf = [0; 3];
        let mut out = Vec::new();
        let err = loop {
            match reader.read(&mut buf) {
                Ok(0) => panic!("the second line is over the ceiling"),
                Ok(n) => out.extend_from_slice(&buf[..n]),
                Err(e) => break e,
            }
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(out.starts_with(b"AAAA\n"));
    }
}
