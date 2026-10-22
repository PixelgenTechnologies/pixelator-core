use std::collections::VecDeque;
use std::fmt;

use crate::common::types::NodeIdx;

/// Returns the mode of `numbers`, breaking ties deterministically by picking
/// the smallest value.
///
/// The function sorts the input slice in-place, then scans contiguous runs of equal
/// values. Because `numbers` is sorted ascending, the first distinct value to achieve
/// the maximum run length is guaranteed to be the smallest tied mode.
#[inline]
pub fn mode_smallest(numbers: &mut [usize]) -> usize {
    assert!(
        !numbers.is_empty(),
        "No numbers to choose from, `numbers` is empty"
    );
    numbers.sort_unstable();
    let mut best_count = 0usize;
    let mut best_n = numbers[0];
    let mut i = 0;
    let len = numbers.len();
    while i < len {
        let val = numbers[i];
        let mut j = i + 1;
        while j < len && numbers[j] == val {
            j += 1;
        }
        let count = j - i;
        if count > best_count {
            best_count = count;
            best_n = val;
        }
        i = j;
    }
    best_n
}

/// Returns the mode of `numbers`, retaining `preferred` if it is among the values
/// tied for the highest frequency, and otherwise breaking ties deterministically
/// by picking the smallest value.
#[inline]
pub fn mode_retain_smallest(numbers: &mut [usize], preferred: usize) -> usize {
    assert!(
        !numbers.is_empty(),
        "No numbers to choose from, `numbers` is empty"
    );
    numbers.sort_unstable();
    let mut best_count = 0usize;
    let mut best_n = numbers[0];
    let mut i = 0;
    let len = numbers.len();
    while i < len {
        let val = numbers[i];
        let mut j = i + 1;
        while j < len && numbers[j] == val {
            j += 1;
        }
        let count = j - i;
        if count > best_count || (count == best_count && val == preferred) {
            best_count = count;
            best_n = val;
        }
        i = j;
    }
    best_n
}

/// Returns the mode (most common value) of a mutable slice of `usize`.
///
/// If `preferred` is `Some(p)` and `p` is tied for the highest frequency, `p` is
/// returned; otherwise ties are broken deterministically by picking the smallest value.
#[inline]
pub fn mode_via_sort(numbers: &mut [usize], preferred: Option<usize>) -> usize {
    match preferred {
        Some(p) => mode_retain_smallest(numbers, p),
        None => mode_smallest(numbers),
    }
}

pub struct NodeQueue {
    queue: VecDeque<NodeIdx>,
    present: Vec<bool>,
}

impl fmt::Debug for NodeQueue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.queue.len() > 20 {
            write!(
                f,
                "NodeQueue(len: {}, front: {:?}, ... , back: {:?})",
                self.queue.len(),
                self.queue.iter().take(3).collect::<Vec<&NodeIdx>>(),
                self.queue.iter().rev().take(3).collect::<Vec<&NodeIdx>>()
            )
        } else {
            write!(
                f,
                "NodeQueue(len: {}, elements: {:?})",
                self.queue.len(),
                self.queue
            )
        }
    }
}
/// The node queue can be used to iterate nodes in order. It has a number of specific requirements:
/// - It assumes all nodes are consecutive integers starting from 0
/// - It assumes all nodes are present in the queue at the time of initialization.
/// - No new nodes will show up during processing, i.e. all indexes 0..n must be known at
///   initialization.
impl NodeQueue {
    pub fn from(vec: Vec<NodeIdx>) -> Self {
        let present = vec![true; vec.len()];

        NodeQueue {
            queue: VecDeque::from(vec),
            present,
        }
    }

    pub fn pop_front(&mut self) -> Option<NodeIdx> {
        let node = self.queue.pop_front();
        if let Some(n) = node {
            self.present[n] = false;
        }
        node
    }

    pub fn push_back(&mut self, node: NodeIdx) {
        if !self.present[node] {
            self.queue.push_back(node);
            self.present[node] = true;
        }
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn contains(&self, node: NodeIdx) -> bool {
        self.present[node]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustc_hash::FxHashSet as HashSet;

    #[test]
    fn test_mode_via_sort_basic() {
        let mut data = vec![1, 2, 2, 3];
        let mode = mode_via_sort(&mut data, None);
        assert_eq!(mode, 2);
        assert_eq!(mode_smallest(&mut data), 2);
    }

    #[test]
    fn test_mode_via_sort_tie_picks_smallest() {
        // Both 1 and 2 appear twice; deterministic tie-break picks smallest (1)
        let mut data = vec![3, 2, 1, 2, 1];
        assert_eq!(mode_smallest(&mut data), 1);
        assert_eq!(mode_via_sort(&mut data, None), 1);
    }

    #[test]
    fn test_mode_retain_smallest_prefers_preferred_on_tie() {
        for preferred in [1, 2, 3] {
            let mut data = vec![3, 1, 2, 1, 2, 3];
            assert_eq!(mode_retain_smallest(&mut data, preferred), preferred);
            assert_eq!(mode_via_sort(&mut data, Some(preferred)), preferred);
        }
    }

    #[test]
    fn test_mode_retain_smallest_falls_back_to_smallest_when_preferred_not_in_tie() {
        let mut data = vec![1, 3, 2, 3, 2];
        assert_eq!(mode_retain_smallest(&mut data, 1), 2);
        assert_eq!(mode_via_sort(&mut data, Some(1)), 2);
        assert_eq!(mode_retain_smallest(&mut data, 99), 2);
    }

    #[test]
    fn test_mode_via_sort_single_element() {
        let mut data = vec![42];
        let mode = mode_via_sort(&mut data, None);
        assert_eq!(mode, 42);
        assert_eq!(mode_smallest(&mut data), 42);
        assert_eq!(mode_retain_smallest(&mut data, 7), 42);
    }

    #[test]
    #[should_panic]
    fn test_mode_via_sort_empty_panics() {
        let mut data: Vec<usize> = vec![];
        let _ = mode_via_sort(&mut data, None);
    }

    #[test]
    #[should_panic]
    fn test_mode_smallest_empty_panics() {
        let mut data: Vec<usize> = vec![];
        let _ = mode_smallest(&mut data);
    }

    #[test]
    #[should_panic]
    fn test_mode_retain_smallest_empty_panics() {
        let mut data: Vec<usize> = vec![];
        let _ = mode_retain_smallest(&mut data, 0);
    }

    #[test]
    fn test_nodequeue_basic_operations() {
        let mut nq = NodeQueue::from(vec![0, 1, 2]);
        assert_eq!(nq.len(), 3);
        assert!(!nq.is_empty());
        assert!(nq.contains(1));

        assert_eq!(nq.pop_front(), Some(0));
        assert!(!nq.contains(0));
        assert_eq!(nq.len(), 2);

        // Already present in queue: should not duplicate.
        nq.push_back(2);
        assert_eq!(nq.len(), 2);

        // Re-enqueueing a popped node should append it.
        nq.push_back(0);
        assert_eq!(nq.len(), 3);
        assert_eq!(nq.pop_front(), Some(1));
        assert_eq!(nq.pop_front(), Some(2));
        assert_eq!(nq.pop_front(), Some(0));
        assert!(nq.is_empty());
    }

    #[test]
    fn test_nodequeue_uniqueness() {
        let mut nq = NodeQueue::from(vec![0, 1]);
        nq.push_back(1); // should not add again
        nq.push_back(0); // should not add again
        assert_eq!(nq.len(), 2);
        let mut seen = HashSet::default();
        while let Some(n) = nq.pop_front() {
            assert!(!seen.contains(&n));
            seen.insert(n);
        }
        assert!(nq.is_empty());
    }

    #[test]
    fn test_nodequeue_from_preserves_order() {
        let mut nq = NodeQueue::from(vec![0, 1, 2, 3]);
        assert_eq!(nq.pop_front(), Some(0));
        assert_eq!(nq.pop_front(), Some(1));
        assert_eq!(nq.pop_front(), Some(2));
        assert_eq!(nq.pop_front(), Some(3));
        assert_eq!(nq.pop_front(), None);
    }

    #[test]
    fn test_nodequeue_reenqueue_after_pop() {
        let mut nq = NodeQueue::from(vec![0, 1]);
        assert_eq!(nq.pop_front(), Some(0));
        assert!(!nq.contains(0));

        nq.push_back(0);
        assert!(nq.contains(0));
        assert_eq!(nq.pop_front(), Some(1));
        assert_eq!(nq.pop_front(), Some(0));
    }

    #[test]
    #[should_panic]
    fn test_nodequeue_contains_out_of_range_panics() {
        let nq = NodeQueue::from(vec![0, 1, 2]);
        let _ = nq.contains(3);
    }

    #[test]
    fn test_nodequeue_empty_invariants() {
        let mut nq = NodeQueue::from(vec![]);
        assert!(nq.is_empty());
        assert_eq!(nq.len(), 0);
        assert_eq!(nq.pop_front(), None);
    }

    #[test]
    #[should_panic]
    fn test_nodequeue_push_back_out_of_range_panics() {
        let mut nq = NodeQueue::from(vec![0, 1, 2]);
        nq.push_back(3);
    }

    #[test]
    fn test_nodequeue_all_nodes_present_at_initialization() {
        let nq = NodeQueue::from(vec![0, 1, 2, 3]);
        assert!(nq.contains(0));
        assert!(nq.contains(1));
        assert!(nq.contains(2));
        assert!(nq.contains(3));
    }
}
