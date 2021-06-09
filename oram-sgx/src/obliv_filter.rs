use crate::align::A64Bytes;
use crate::bitonic_sort::BiotonicSort;
use timing_shield::{TpBool, TpCondSwap, TpEq, TpOrd, TpU32, TpU8};

pub fn obliv_filter<const N: usize>(
    bitmap: &[TpBool],
    data: &[A64Bytes<N>],
    filter_capacity: u32,
) -> (Vec<A64Bytes<N>>, TpU32) {
    assert_eq!(bitmap.len(), data.len());
    let mut current_len = TpU32::protect(0);
    let capacity = filter_capacity as usize;
    let mut filtered = vec![A64Bytes::<N>::default(); capacity];
    for (bitmap_chunk, data_chunk) in bitmap.chunks(capacity).zip(data.chunks(capacity)) {
        let mut obliv_items = to_obliv_items(bitmap_chunk, data_chunk, current_len, capacity);
        BiotonicSort::sort(&mut obliv_items, true);
        for (tar, (b, src)) in filtered
            .iter_mut()
            .zip(obliv_items.into_iter().map(|v| v.split()))
        {
            tar.cmov(b, &src);
            current_len = b.select(current_len + 1, current_len);
        }
    }
    (filtered, current_len)
}

fn assign_bitmap(bitmap: &[TpBool], true_start: TpU32) -> Vec<TpU8> {
    let zero = TpU8::protect(0);
    let one = TpU8::protect(1);
    let two = TpU8::protect(2);

    let mut n_false_left = true_start;
    let mut out_of_padding_false = n_false_left.tp_eq(&0);
    bitmap
        .iter()
        .map(|&b| {
            let out = b.select(one, out_of_padding_false.select(two, zero));
            n_false_left = b.select(
                n_false_left,
                out_of_padding_false.select(n_false_left, n_false_left - 1),
            );
            out_of_padding_false = n_false_left.tp_eq(&0);
            out
        })
        .collect()
}

struct OblivFilterItem<const N: usize> {
    num: TpU8,
    data: Box<A64Bytes<N>>,
}
impl<const N: usize> OblivFilterItem<N> {
    pub fn split(self) -> (TpBool, Box<A64Bytes<N>>) {
        ((self.num & 1).tp_not_eq(&0), self.data)
    }
}

impl<const N: usize> TpOrd for OblivFilterItem<N> {
    #[inline]
    fn tp_lt(&self, other: &Self) -> TpBool {
        self.num.tp_lt(&other.num)
    }

    #[inline]
    fn tp_lt_eq(&self, other: &Self) -> TpBool {
        self.num.tp_lt_eq(&other.num)
    }

    #[inline]
    fn tp_gt(&self, other: &Self) -> TpBool {
        self.num.tp_gt(&other.num)
    }

    #[inline]
    fn tp_gt_eq(&self, other: &Self) -> TpBool {
        self.num.tp_gt_eq(&other.num)
    }
}

impl<const N: usize> TpCondSwap for OblivFilterItem<N> {
    fn tp_cond_swap(condition: TpBool, a: &mut Self, b: &mut Self) {
        condition.cond_swap(&mut a.num, &mut b.num);
        condition.cond_swap(&mut *a.data, &mut *b.data);
    }
}

fn to_obliv_items<const N: usize>(
    bitmap: &[TpBool],
    all_data: &[A64Bytes<N>],
    true_start: TpU32,
    resize: usize,
) -> Vec<OblivFilterItem<N>> {
    assert_eq!(bitmap.len(), all_data.len());
    assert!(resize >= all_data.len());
    let nummap = if all_data.len() < resize {
        let mut bitmap = bitmap.to_vec();
        bitmap.resize(resize, TpBool::protect(false));
        assign_bitmap(&bitmap, true_start)
    } else {
        assign_bitmap(bitmap, true_start)
    };
    let blank;
    let data_iter: Box<dyn Iterator<Item = &A64Bytes<N>>> = if all_data.len() < resize {
        blank = A64Bytes::<N>::default();
        Box::new(all_data.iter().chain(std::iter::repeat(&blank)))
    } else {
        Box::new(all_data.iter())
    };
    nummap
        .into_iter()
        .zip(data_iter)
        .map(|(num, data)| OblivFilterItem {
            num,
            data: Box::new(data.clone()),
        })
        .collect()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn assign_bitmap_test() {
        use rand::{Rng, RngCore};
        let mut rng = rand::thread_rng();
        let n = 1000;
        let mut rand_bitmap = vec![0u8; n];
        rng.fill_bytes(&mut rand_bitmap[..]);
        let ref_rand_bitmap = rand_bitmap
            .into_iter()
            .map(|i| (i & 1) != 0)
            .collect::<Vec<_>>();
        let num_trues = ref_rand_bitmap.iter().filter(|&&i| i).count();
        let rand_bitmap = ref_rand_bitmap
            .iter()
            .map(|&i| TpBool::protect(i))
            .collect::<Vec<_>>();
        let true_start = rng.gen_range((n / 4)..(n / 3)) as u32;
        let num_map = assign_bitmap(&rand_bitmap, TpU32::protect(true_start));
        let mut num_map = num_map.into_iter().map(|i| i.expose()).collect::<Vec<_>>();
        num_map.sort_unstable();
        let true_start = true_start as usize;
        assert!(num_map[0..true_start].into_iter().all(|&i| i == 0));
        assert!(num_map[true_start..(true_start + num_trues)]
            .into_iter()
            .all(|&i| i == 1));
        assert!(num_map[(true_start + num_trues)..]
            .into_iter()
            .all(|&i| i == 2));
    }

    #[test]
    fn bitonic_sort_obliv_items() {
        use rand::{Rng, RngCore};
        let mut rng = rand::thread_rng();
        let n = 1000;
        let mut rand_bitmap = vec![0u8; n];
        rng.fill_bytes(&mut rand_bitmap[..]);
        let ref_rand_bitmap = rand_bitmap
            .into_iter()
            .map(|i| (i & 1) != 0)
            .collect::<Vec<_>>();
        let num_trues = ref_rand_bitmap.iter().filter(|&&i| i).count();
        let data = ref_rand_bitmap
            .iter()
            .map(|&b| {
                if b {
                    A64Bytes::<1>::from_slice(&[1])
                } else {
                    A64Bytes::<1>::from_slice(&[0])
                }
            })
            .collect::<Vec<_>>();
        let rand_bitmap = ref_rand_bitmap
            .iter()
            .map(|&i| TpBool::protect(i))
            .collect::<Vec<_>>();
        let true_start = rng.gen_range((n / 4)..(n / 3)) as u32;

        let mut obliv_items = to_obliv_items(
            &rand_bitmap,
            &data,
            TpU32::protect(true_start),
            rand_bitmap.len(),
        );
        crate::bitonic_sort::BiotonicSort::sort(&mut obliv_items, true);
        let obliv_items = obliv_items
            .into_iter()
            .map(|v| v.split())
            .collect::<Vec<_>>();
        for (b, d) in &obliv_items {
            println!("{}", b.expose());
            assert_eq!(b.expose() as u8, d.as_slice()[0]);
        }
        let true_start = true_start as usize;
        assert!(obliv_items[0..true_start]
            .into_iter()
            .all(|i| !i.0.expose()));
        assert!(obliv_items[true_start..(true_start + num_trues)]
            .into_iter()
            .all(|i| i.0.expose()));
        assert!(obliv_items[(true_start + num_trues)..]
            .into_iter()
            .all(|i| !i.0.expose()));
    }

    #[test]
    fn obliv_filter_test() {
        use rand::prelude::SliceRandom;
        let mut rng = rand::thread_rng();
        let k = 10;
        let n = 1000;
        let data = (0..n)
            .map(|i| A64Bytes::<1>::from_slice(&[(i % 256) as u8]))
            .collect::<Vec<_>>();
        let selected = (0..n)
            .collect::<Vec<_>>()
            .choose_multiple(&mut rng, k)
            .cloned()
            .collect::<Vec<_>>();
        let mut bitmap = vec![false; n];
        for i in selected {
            bitmap[i] = true;
        }

        let mut ref_selected_data = data
            .iter()
            .cloned()
            .zip(bitmap.iter())
            .filter(|(_, &b)| b)
            .map(|(v, _)| v)
            .collect::<Vec<_>>();
        ref_selected_data.sort_unstable_by_key(|v| v.as_slice()[0]);

        let obliv_bitmap = bitmap
            .into_iter()
            .map(|b| TpBool::protect(b))
            .collect::<Vec<_>>();
        let (mut filtered, len) = obliv_filter(&obliv_bitmap, &data, (k + 2) as u32);
        assert_eq!(len.expose(), k as u32);
        filtered.resize(k, A64Bytes::<1>::default());
        filtered.sort_unstable_by_key(|v| v.as_slice()[0]);
        assert!(ref_selected_data
            .into_iter()
            .zip(filtered.into_iter())
            .all(|(a, b)| a.as_slice()[0] == b.as_slice()[0]));
    }
}
