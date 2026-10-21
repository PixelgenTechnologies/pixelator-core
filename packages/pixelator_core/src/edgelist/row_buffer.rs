use std::mem;
use std::sync::Arc;

use arrow::array::{ArrayRef, DictionaryArray, Int32Array, StringArray, UInt64Array};
use arrow::datatypes::{DataType, Field, Int32Type, Schema};
use arrow::error::ArrowError;
use arrow::record_batch::RecordBatch;

use crate::common::Panel;

#[derive(Clone)]
pub struct RowBuffer {
    /// Marker indices for `pid-1`; each entry is a key into `dict` and is
    /// emitted as a dictionary-encoded (VARCHAR) `marker_1` column.
    buf_marker_idx1: Vec<i32>,
    /// Marker indices for `pid-2`; each entry is a key into `dict` and is
    /// emitted as a dictionary-encoded (VARCHAR) `marker_2` column.
    buf_marker_idx2: Vec<i32>,
    buf_umi1: Vec<u64>,
    buf_umi2: Vec<u64>,
    /// Shared dictionary values: the panel's marker ids in file order, indexed
    /// by the `buf_marker_idx*` keys. Built once in `new` and reused (as a cheap
    /// `Arc` clone) for every flushed batch.
    dict: ArrayRef,
}

impl RowBuffer {
    /// Creates a buffer pre-sized to `capacity` rows.
    ///
    /// The `panel` is only borrowed to snapshot its marker ids into the shared
    /// dictionary values array. The marker indices later pushed via
    /// [`push`](Self::push) must be positions in this same panel's `markers`
    /// (which is what [`Panel::marker_index_seq1`] and friends return).
    pub fn new(capacity: usize, panel: &Panel) -> Self {
        let dict: ArrayRef = Arc::new(StringArray::from_iter_values(
            panel.markers().iter().map(|m| m.marker_id.as_str()),
        ));
        Self {
            buf_marker_idx1: Vec::with_capacity(capacity),
            buf_marker_idx2: Vec::with_capacity(capacity),
            buf_umi1: Vec::with_capacity(capacity),
            buf_umi2: Vec::with_capacity(capacity),
            dict,
        }
    }

    pub fn push(&mut self, marker_idx1: i32, marker_idx2: i32, umi1: u64, umi2: u64) {
        self.buf_marker_idx1.push(marker_idx1);
        self.buf_marker_idx2.push(marker_idx2);
        self.buf_umi1.push(umi1);
        self.buf_umi2.push(umi2);
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.buf_marker_idx1.len() == 0
    }

    pub(crate) fn len(&self) -> usize {
        self.buf_marker_idx1.len()
    }

    pub(crate) fn schema() -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new(
                "marker_1",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                false,
            ),
            Field::new(
                "marker_2",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                false,
            ),
            Field::new("umi1", DataType::UInt64, false),
            Field::new("umi2", DataType::UInt64, false),
        ]))
    }

    /// Move the buffered columns into an Arrow `RecordBatch` and reset the buffers, preserving each
    /// vector's previous capacity for the next batch.
    ///
    /// All four columns hand their `Vec`s to Arrow zero-copy: `Buffer::from_vec` adopts each
    /// vector's existing allocation rather than copying elements, so both the `i32` marker keys and
    /// the `u64` UMIs are transferred without a per-row copy. The marker columns are additionally
    /// built as `DictionaryArray`s that share a single `Arc`'d dictionary of marker ids
    /// (`self.dict.clone()` is just a refcount bump)
    ///
    /// Parquet stores the marker columns as RLE_DICTIONARY-encoded VARCHAR. The reset leaves
    /// freshly allocated buffers, pre-sized to avoid re-growing while filling the next batch.
    pub fn flush(&mut self) -> Result<RecordBatch, ArrowError> {
        let marker_idx1 = take_buffer(&mut self.buf_marker_idx1);
        let marker_idx2 = take_buffer(&mut self.buf_marker_idx2);
        let umi1 = take_buffer(&mut self.buf_umi1);
        let umi2 = take_buffer(&mut self.buf_umi2);

        let marker1 = DictionaryArray::<Int32Type>::try_new(
            Int32Array::from(marker_idx1),
            self.dict.clone(),
        )?;
        let marker2 = DictionaryArray::<Int32Type>::try_new(
            Int32Array::from(marker_idx2),
            self.dict.clone(),
        )?;

        let columns: Vec<ArrayRef> = vec![
            Arc::new(marker1),
            Arc::new(marker2),
            Arc::new(UInt64Array::from(umi1)),
            Arc::new(UInt64Array::from(umi2)),
        ];

        RecordBatch::try_new(Self::schema(), columns)
    }
}

/// Reset a column buffer, returning its contents and leaving a fresh buffer
/// with the same capacity for the next batch.
fn take_buffer<T>(buf: &mut Vec<T>) -> Vec<T> {
    let cap = buf.capacity();
    mem::replace(buf, Vec::with_capacity(cap))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fixtures::proxiome_v2_panel;

    /// Builds the expected `DictionaryArray` for a marker column from a set of
    /// keys and the panel used to construct the buffer.
    fn expected_marker_column(keys: Vec<i32>, panel: &Panel) -> DictionaryArray<Int32Type> {
        let values: ArrayRef = Arc::new(StringArray::from_iter_values(
            panel.markers().iter().map(|m| m.marker_id.as_str()),
        ));
        DictionaryArray::<Int32Type>::try_new(Int32Array::from(keys), values).unwrap()
    }

    #[test]
    fn new_preallocates_all_columns() {
        let cap = 128;
        let rb = RowBuffer::new(cap, &proxiome_v2_panel());

        // `with_capacity` guarantees "at least cap", so assert the lower bound.
        assert!(rb.buf_marker_idx1.capacity() >= cap);
        assert!(rb.buf_marker_idx2.capacity() >= cap);
        assert!(rb.buf_umi1.capacity() >= cap);
        assert!(rb.buf_umi2.capacity() >= cap);

        assert_eq!(rb.len(), 0);
        assert!(rb.is_empty());
    }

    #[test]
    fn push_appends_each_column_in_order() {
        let mut rb = RowBuffer::new(4, &proxiome_v2_panel());
        rb.push(1, 3, 11, 21);
        rb.push(2, 4, 12, 22);
        rb.push(3, 5, 13, 23);

        assert_eq!(rb.len(), 3);

        assert_eq!(rb.buf_marker_idx1, vec![1, 2, 3]);
        assert_eq!(rb.buf_marker_idx2, vec![3, 4, 5]);
        assert_eq!(rb.buf_umi1, vec![11, 12, 13]);
        assert_eq!(rb.buf_umi2, vec![21, 22, 23]);
    }

    #[test]
    fn flush_produces_expected_batch() {
        let panel = proxiome_v2_panel();
        let mut rb = RowBuffer::new(4, &panel);
        rb.push(0, 2, 11, 21);
        rb.push(1, 3, 12, 22);

        let batch = rb.flush().expect("flush");

        let expected = RecordBatch::try_new(
            RowBuffer::schema(),
            vec![
                Arc::new(expected_marker_column(vec![0, 1], &panel)) as ArrayRef,
                Arc::new(expected_marker_column(vec![2, 3], &panel)),
                Arc::new(UInt64Array::from(vec![11u64, 12])),
                Arc::new(UInt64Array::from(vec![21u64, 22])),
            ],
        )
        .unwrap();

        assert_eq!(batch, expected);
    }

    #[test]
    fn flush_resets_buffers_and_keeps_capacity() {
        let mut rb = RowBuffer::new(16, &proxiome_v2_panel());
        rb.push(0, 1, 1, 2);

        let cap_before = rb.buf_marker_idx1.capacity();
        let _ = rb.flush().unwrap();

        assert_eq!(rb.len(), 0);
        assert_eq!(rb.buf_marker_idx2.len(), 0);
        assert_eq!(rb.buf_umi1.len(), 0);
        assert_eq!(rb.buf_umi2.len(), 0);

        assert!(rb.buf_marker_idx1.capacity() >= cap_before);
    }

    #[test]
    fn flush_does_not_carry_over_previous_rows() {
        let panel = proxiome_v2_panel();
        let mut rb = RowBuffer::new(4, &panel);
        rb.push(5, 1, 11, 21);
        let first = rb.flush().unwrap();
        assert_eq!(first.num_rows(), 1);

        rb.push(6, 2, 12, 22);
        let second = rb.flush().unwrap();
        assert_eq!(second.num_rows(), 1);
        // and the value is the *new* one, not the old
        let marker1 = second
            .column(0)
            .as_any()
            .downcast_ref::<DictionaryArray<Int32Type>>()
            .unwrap();
        assert_eq!(marker1.keys().value(0), 6);
    }

    #[test]
    fn flush_on_empty_returns_empty_batch() {
        let mut rb = RowBuffer::new(0, &proxiome_v2_panel());
        let batch = rb.flush().unwrap();
        assert_eq!(batch.num_rows(), 0);
        assert_eq!(batch.num_columns(), 4);
        assert_eq!(batch.schema(), RowBuffer::schema());
    }

    #[test]
    fn schema_matches_output_contract() {
        let schema = RowBuffer::schema();
        let fields = schema.fields();
        assert_eq!(fields.len(), 4);

        let dict_type = DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        let expected = [
            ("marker_1", dict_type.clone()),
            ("marker_2", dict_type),
            ("umi1", DataType::UInt64),
            ("umi2", DataType::UInt64),
        ];
        for (field, (name, dtype)) in fields.iter().zip(expected) {
            assert_eq!(field.name(), name);
            assert_eq!(field.data_type(), &dtype);
            assert!(!field.is_nullable());
        }
    }
}
