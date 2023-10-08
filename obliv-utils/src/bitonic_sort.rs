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

fn merge<T: TpOrd + TpCondSwap>(items: &mut [T], dir: bool) {
    let n = items.len();
    if n > 1 {
        let m = n.next_power_of_two() >> 1;
        for i in 0..(n - m) {
            let cond = items[i].tp_gt_eq(&items[i + m]).tp_eq(&dir);
            cond_swap_in_slice(cond, items, i, i + m);
        }
        merge(&mut items[..m], dir);
        merge(&mut items[m..n], dir);
    }
}

fn cond_swap_in_slice<T: TpCondSwap>(cond: TpBool, slice: &mut [T], i: usize, j: usize) {
    assert!(j < slice.len());
    assert!(i < j);
    let (a, b) = slice.split_at_mut(i + 1);
    let a = &mut a[i];
    let b = &mut b[j - (i + 1)];
    cond.cond_swap(a, b);
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn bitonic_sort_test() {
        use rand::RngCore;
        let mut rng = rand::thread_rng();
        let n = 1000;
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
}
