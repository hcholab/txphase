use crate::vec::{rl_cap, OblivVec};
use timing_shield::{TpCondSwap, TpOrd};

pub struct OblivMinHeap<T>
where
    [(); rl_cap::<T>()]:,
{
    inner: OblivVec<T>,
    heap_size: usize,
}

impl<T> OblivMinHeap<T>
where
    [(); rl_cap::<T>()]:,
    T: TpOrd + TpCondSwap,
{
    pub fn with_heap_size(heap_size: usize) -> Self {
        Self {
            inner: OblivVec::<T>::with_capacity(heap_size),
            heap_size,
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn insert(&mut self, mut item: T) {
        if self.inner.len() < self.heap_size {
            self.inner.push(item);
        } else {
            let last = self.inner.last_mut().unwrap();
            let cond = item.tp_lt(last);
            cond.cond_swap(last, &mut item);
        }
        self.inner.bubble_sort_last();
    }

    pub fn into_iter(self) -> impl Iterator<Item = T> {
        self.inner.into_iter()
    }

    pub fn into_obliv_vec(self) -> OblivVec<T> {
        self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obliv_min_heap() {
        let ref_numbers = (0..200).rev().collect::<Vec<_>>();
        let ref_numbers = ref_numbers
            .into_iter()
            .map(|v| timing_shield::TpI64::protect(v))
            .collect::<Vec<_>>();
        let mut min_heap = OblivMinHeap::with_heap_size(35);
        for i in ref_numbers {
            min_heap.insert(i);
        }
        let test_numbers = min_heap.into_iter().map(|v| v.expose()).collect::<Vec<_>>();
        assert_eq!((0..35).collect::<Vec<_>>(), test_numbers);
    }
}
