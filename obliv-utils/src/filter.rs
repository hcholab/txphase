use timing_shield::{TpBool, TpCondSwap, TpOrd, TpU32, TpU64};

struct SortItem<T>((TpU64, T));

impl<T> TpOrd for SortItem<T> {
    fn tp_lt(&self, other: &Self) -> TpBool {
        self.0 .0.tp_lt(&other.0 .0)
    }
    fn tp_lt_eq(&self, other: &Self) -> TpBool {
        self.0 .0.tp_lt_eq(&other.0 .0)
    }
    fn tp_gt_eq(&self, other: &Self) -> TpBool {
        self.0 .0.tp_gt_eq(&other.0 .0)
    }
    fn tp_gt(&self, other: &Self) -> TpBool {
        self.0 .0.tp_gt(&other.0 .0)
    }
}

impl<T: TpCondSwap> TpCondSwap for SortItem<T> {
    fn tp_cond_swap(cond: TpBool, a: &mut Self, b: &mut Self) {
        TpU64::tp_cond_swap(cond, &mut a.0 .0, &mut b.0 .0);
        T::tp_cond_swap(cond, &mut a.0 .1, &mut b.0 .1);
    }
}

pub fn obliv_filter<T: TpCondSwap>(
    filter: impl Iterator<Item = TpBool>,
    items: impl Iterator<Item = T>,
    max_n_filtered: usize,
) -> (Vec<T>, TpU32) {
    let mut count_1 = TpU64::protect(0);
    let mut count_0 = TpU64::protect(0);
    let mut sort_items = filter
        .zip(items)
        .map(|(b, d)| {
            count_0 += (!b).as_u64();
            count_1 += b.as_u64();
            let sorting_order = b.select(count_1, count_0 << 32);
            SortItem((sorting_order, d))
        })
        .collect::<Vec<_>>();

    crate::bitonic_sort::bitonic_sort_top_s(&mut sort_items, max_n_filtered, true);

    let target = sort_items
        .into_iter()
        .map(|SortItem((_, d))| d)
        .collect::<Vec<_>>();
    (target, count_1.as_u32())
}
