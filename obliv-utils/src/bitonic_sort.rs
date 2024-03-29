use timing_shield::{TpBool, TpCondSwap, TpEq, TpOrd};

// ref: https://github.com/flakusha/sorting_rs/blob/master/src/bitonic_sort.rs
pub fn bitonic_sort<T: TpOrd + TpCondSwap>(items: &mut [T], ascending: bool) {
    let n = items.len();
    if n > 1 {
        let dir = ascending;
        let m = n / 2;
        bitonic_sort(&mut items[..m], !dir);
        bitonic_sort(&mut items[m..n], dir);
        merge(items, dir)
    }
}

// ref: https://github.com/flakusha/sorting_rs/blob/master/src/bitonic_sort.rs
pub fn bitonic_sort_top_s<T: TpOrd + TpCondSwap>(items: &mut [T], s: usize, ascending: bool) {
    let n = items.len();
    assert!(s <= n);
    if n > 1 {
        let dir = ascending;
        let m = n / 2;
        let s = s.min(m);
        bitonic_sort(&mut items[..m], !dir);
        bitonic_sort(&mut items[m..n], dir);
        merge(&mut items[m - s..m + s], dir);
        items.rotate_left(m - s);
    }
}

fn merge<T: TpOrd + TpCondSwap>(items: &mut [T], dir: bool) {
    let n = items.len();
    if n > 1 {
        let m = n.next_power_of_two() >> 1;
        let (a, b) = items.split_at_mut(m);
        for (a, b) in a.iter_mut().take(n - m).zip(b.iter_mut().take(n - m)) {
            let cond = a.tp_gt_eq(b).tp_eq(&dir);
            cond.cond_swap(a, b);
        }
        merge(&mut items[..m], dir);
        merge(&mut items[m..n], dir);
    }
}

pub fn bitonic_sort_by<T>(
    items: &mut [T],
    ascending: bool,
    gt_eq: fn(&T, &T) -> TpBool,
    cond_swap: fn(TpBool, &mut T, &mut T),
) {
    let n = items.len();
    if n > 1 {
        let dir = ascending;
        let m = n / 2;
        bitonic_sort_by(&mut items[..m], !dir, gt_eq, cond_swap);
        bitonic_sort_by(&mut items[m..n], dir, gt_eq, cond_swap);
        merge_by(items, dir, gt_eq, cond_swap)
    }
}

fn merge_by<T>(
    items: &mut [T],
    dir: bool,
    gt_eq: fn(&T, &T) -> TpBool,
    cond_swap: fn(TpBool, &mut T, &mut T),
) {
    let n = items.len();
    if n > 1 {
        let m = n.next_power_of_two() >> 1;
        let (a, b) = items.split_at_mut(m);
        for (a, b) in a.iter_mut().take(n - m).zip(b.iter_mut().take(n - m)) {
            let cond = gt_eq(a, b).tp_eq(&dir);
            cond_swap(cond, a, b);
        }
        merge_by(&mut items[..m], dir, gt_eq, cond_swap);
        merge_by(&mut items[m..n], dir, gt_eq, cond_swap);
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn bitonic_sort_test() {
        use rand::RngCore;
        let mut rng = rand::thread_rng();
        let n = 1000;

        {
            let mut ref_nums = (0..n).map(|_| rng.next_u32()).collect::<Vec<_>>();
            let mut nums = ref_nums
                .iter()
                .map(|&v| timing_shield::TpU32::protect(v))
                .collect::<Vec<_>>();
            bitonic_sort(&mut nums, true);
            ref_nums.sort_unstable();
            let nums = nums.into_iter().map(|v| v.expose()).collect::<Vec<_>>();
            assert_eq!(ref_nums, nums);
        }

        {
            let mut ref_nums = (0..n).map(|_| rng.next_u32()).collect::<Vec<_>>();
            let mut nums = ref_nums
                .iter()
                .map(|&v| timing_shield::TpU32::protect(v))
                .collect::<Vec<_>>();
            bitonic_sort_top_s(&mut nums, 100, true);
            ref_nums.sort_unstable();
            let ref_nums = ref_nums.into_iter().take(100).collect::<Vec<_>>();
            let nums = nums
                .into_iter()
                .take(100)
                .map(|v| v.expose())
                .collect::<Vec<_>>();
            assert_eq!(ref_nums, nums);
        }
    }
}
