use std::sync::Arc;

use arrow::array::{ArrayRef, DictionaryArray, Int32Array, StringArray};
use arrow::buffer::{Buffer, NullBuffer, OffsetBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, Field, Int32Type, Schema};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;
use paraseq::Record;

use super::extract::DropReason;
use super::read_layout::Read;

/// Column buffers for the reads the extractor discarded, flushed into the
/// optional dropped-reads Parquet that sits next to the edgelist.
///
/// One row per input molecule, exactly as it appeared in the FASTQ: a single-end
/// read fills `sequence`/`quality` and leaves the mate columns null; a pair puts
/// R1 in `sequence`/`quality` and R2 in `sequence_2`/`quality_2`. `reason` is
/// the [`DropReason`] that fired first.
#[derive(Clone)]
pub struct DroppedReadBuffer {
    read_name: StringColumn,
    sequence: StringColumn,
    quality: StringColumn,
    sequence_2: StringColumn,
    quality_2: StringColumn,
    /// Keys into `reason_dict`, i.e. [`DropReason::index`].
    reason: Vec<i32>,
    /// The names of [`DropReason::ALL`], shared by every flushed batch.
    reason_dict: ArrayRef,
}

impl DroppedReadBuffer {
    /// Creates a buffer pre-sized to `capacity` rows.
    pub fn new(capacity: usize) -> Self {
        Self {
            read_name: StringColumn::with_capacity(capacity),
            sequence: StringColumn::with_capacity(capacity),
            quality: StringColumn::with_capacity(capacity),
            sequence_2: StringColumn::with_capacity(capacity),
            quality_2: StringColumn::with_capacity(capacity),
            reason: Vec::with_capacity(capacity),
            reason_dict: Arc::new(StringArray::from_iter_values(
                DropReason::ALL.iter().map(DropReason::as_str),
            )),
        }
    }

    pub fn push<R: Record>(&mut self, read: Read<'_, R>, reason: DropReason) {
        let (first, mate) = match read {
            Read::SingleForward(r) | Read::SingleReverse(r) => (r, None),
            Read::Paired { r1, r2 } => (r1, Some(r2)),
        };
        self.read_name.push(Some(read_name(first.id())));
        self.sequence.push(Some(first.seq().as_ref()));
        self.quality.push(first.qual());
        self.sequence_2.push(mate.map(|r| r.seq()).as_deref());
        self.quality_2.push(mate.and_then(|r| r.qual()));
        self.reason.push(reason.index() as i32);
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.reason.is_empty()
    }

    pub(crate) fn len(&self) -> usize {
        self.reason.len()
    }

    pub(crate) fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("read_name", DataType::Utf8, false),
            Field::new("sequence", DataType::Utf8, false),
            // Nullable: paraseq also reads FASTA, which has no qualities.
            Field::new("quality", DataType::Utf8, true),
            Field::new("sequence_2", DataType::Utf8, true),
            Field::new("quality_2", DataType::Utf8, true),
            Field::new(
                "reason",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                false,
            ),
        ]))
    }

    /// Moves the buffered rows into an Arrow `RecordBatch` and resets the buffer.
    ///
    /// Fails if a read name, sequence or quality string is not valid UTF-8.
    pub fn flush(&mut self) -> Result<RecordBatch, ArrowError> {
        let reason = DictionaryArray::<Int32Type>::try_new(
            Int32Array::from(std::mem::take(&mut self.reason)),
            self.reason_dict.clone(),
        )?;
        let columns: Vec<ArrayRef> = vec![
            Arc::new(self.read_name.finish()?),
            Arc::new(self.sequence.finish()?),
            Arc::new(self.quality.finish()?),
            Arc::new(self.sequence_2.finish()?),
            Arc::new(self.quality_2.finish()?),
            Arc::new(reason),
        ];
        RecordBatch::try_new(Self::schema(), columns)
    }
}

/// The read name is the header up to its first whitespace, as samtools and most
/// tools define it; the rest is a free-form comment (and differs between mates).
fn read_name(id: &[u8]) -> &[u8] {
    id.split(u8::is_ascii_whitespace).next().unwrap_or(id)
}

/// A growable Arrow `Utf8` column.
#[derive(Clone)]
struct StringColumn {
    values: Vec<u8>,
    /// `offsets[i]..offsets[i + 1]` delimits row `i` in `values`; always starts
    /// with a leading 0, so it holds one more entry than there are rows.
    offsets: Vec<i32>,
    validity: Vec<bool>,
    has_nulls: bool,
}

impl StringColumn {
    fn with_capacity(rows: usize) -> Self {
        let mut offsets = Vec::with_capacity(rows + 1);
        offsets.push(0);
        Self {
            values: Vec::new(),
            offsets,
            validity: Vec::with_capacity(rows),
            has_nulls: false,
        }
    }

    fn push(&mut self, value: Option<&[u8]>) {
        if let Some(bytes) = value {
            self.values.extend_from_slice(bytes);
        }
        self.has_nulls |= value.is_none();
        self.validity.push(value.is_some());
        // An `i32` offset caps a batch at 2 GiB of text; flushing every
        // `DROPPED_BATCH_THRESHOLD` rows keeps short reads far below that, and
        // overflow is still caught by the offset validation in `finish`.
        self.offsets.push(self.values.len() as i32);
    }

    /// Hands the buffered values to Arrow (zero-copy) and resets the column.
    fn finish(&mut self) -> Result<StringArray, ArrowError> {
        let rows = self.validity.len();
        let values = std::mem::take(&mut self.values);
        let offsets = std::mem::replace(&mut self.offsets, {
            let mut fresh = Vec::with_capacity(rows + 1);
            fresh.push(0);
            fresh
        });
        let validity = std::mem::replace(&mut self.validity, Vec::with_capacity(rows));
        let nulls = std::mem::take(&mut self.has_nulls).then(|| NullBuffer::from(validity));

        StringArray::try_new(
            OffsetBuffer::new(ScalarBuffer::from(offsets)),
            Buffer::from_vec(values),
            nulls,
        )
    }
}

#[cfg(test)]
mod tests {
    use arrow::array::Array;

    use super::*;
    use crate::test_fixtures::OwnedRecord;

    fn record(id: &str, seq: &str, qual: &str) -> OwnedRecord {
        OwnedRecord {
            id: id.as_bytes().to_vec(),
            seq: seq.as_bytes().to_vec(),
            qual: qual.as_bytes().to_vec(),
        }
    }

    fn string_column<'a>(batch: &'a RecordBatch, name: &str) -> &'a StringArray {
        batch
            .column_by_name(name)
            .expect("column present")
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("utf8 column")
    }

    fn reasons(batch: &RecordBatch) -> Vec<String> {
        let dict = batch
            .column_by_name("reason")
            .expect("reason column")
            .as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>()
            .expect("dictionary-encoded reason");
        let names = dict
            .values()
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("string dictionary values");
        dict.keys()
            .iter()
            .map(|k| names.value(k.expect("non-null key") as usize).to_string())
            .collect()
    }

    #[test]
    fn single_end_read_leaves_mate_columns_null() {
        let r = record("read0 1:N:0:ACGT", "ACGTN", "IIII#");
        let mut buf = DroppedReadBuffer::new(4);
        buf.push(Read::SingleForward(&&r), DropReason::HasN);

        let batch = buf.flush().expect("flush");
        assert_eq!(batch.schema(), DroppedReadBuffer::schema());
        assert_eq!(batch.num_rows(), 1);
        assert_eq!(string_column(&batch, "read_name").value(0), "read0");
        assert_eq!(string_column(&batch, "sequence").value(0), "ACGTN");
        assert_eq!(string_column(&batch, "quality").value(0), "IIII#");
        assert!(string_column(&batch, "sequence_2").is_null(0));
        assert!(string_column(&batch, "quality_2").is_null(0));
        assert_eq!(reasons(&batch), ["has_n"]);
    }

    #[test]
    fn paired_read_keeps_r1_name_and_both_mates() {
        let r1 = record("pair0 1:N:0", "AAAA", "FFFF");
        let r2 = record("pair0 2:N:0", "CCCCC", "#####");
        let mut buf = DroppedReadBuffer::new(4);
        buf.push(
            Read::Paired { r1: &&r1, r2: &&r2 },
            DropReason::Pid2NotInPanel,
        );

        let batch = buf.flush().expect("flush");
        assert_eq!(string_column(&batch, "read_name").value(0), "pair0");
        assert_eq!(string_column(&batch, "sequence").value(0), "AAAA");
        assert_eq!(string_column(&batch, "quality").value(0), "FFFF");
        assert_eq!(string_column(&batch, "sequence_2").value(0), "CCCCC");
        assert_eq!(string_column(&batch, "quality_2").value(0), "#####");
        assert_eq!(reasons(&batch), ["pid2_not_in_panel"]);
    }

    #[test]
    fn flush_resets_the_buffer_for_the_next_batch() {
        let a = record("a", "AC", "II");
        let b = record("b", "GTT", "III");
        let mut buf = DroppedReadBuffer::new(1);

        buf.push(Read::SingleReverse(&&a), DropReason::TooShort);
        buf.push(Read::SingleReverse(&&b), DropReason::LowQuality);
        assert_eq!(buf.len(), 2);
        let first = buf.flush().expect("first flush");
        assert!(buf.is_empty());

        buf.push(Read::SingleReverse(&&b), DropReason::LbsOverlap);
        let second = buf.flush().expect("second flush");

        let names = string_column(&first, "read_name");
        assert_eq!((names.value(0), names.value(1)), ("a", "b"));
        assert_eq!(string_column(&first, "sequence").value(1), "GTT");
        assert_eq!(reasons(&first), ["too_short", "low_quality"]);

        assert_eq!(second.num_rows(), 1);
        assert_eq!(string_column(&second, "sequence").value(0), "GTT");
        assert_eq!(reasons(&second), ["lbs_overlap"]);
    }

    #[test]
    fn read_name_stops_at_the_first_whitespace() {
        assert_eq!(read_name(b"name comment"), b"name");
        assert_eq!(read_name(b"name\tcomment"), b"name");
        assert_eq!(read_name(b"name"), b"name");
        assert_eq!(read_name(b""), b"");
    }
}
