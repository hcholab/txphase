use crate::min_heap::OblivMinHeap;
use crate::vec::{rl_cap, OblivVec};
use timing_shield::{TpCondSwap, TpOrd, TpU8};

pub fn select_top_s<T>(s: usize, ranks: &[T]) -> OblivVec<T>
where
    [(); rl_cap::<T>()]:,
    T: TpOrd + TpCondSwap + Clone,
{
    let mut top_s = OblivMinHeap::with_heap_size(s);
    for rank in ranks {
        top_s.insert(rank.clone())
    }
    top_s.into_obliv_vec()
}

// assume the ranks are sorted
pub fn merge_top_s<T>(s: usize, ranks_a: &OblivVec<T>, ranks_b: &OblivVec<T>) -> OblivVec<T>
where
    [(); rl_cap::<T>()]:,
    T: Clone + TpOrd + TpCondSwap,
{
    let mut top_s = OblivVec::with_capacity(s);

    let mut p_a = TpU8::protect(0);
    let mut p_b = TpU8::protect(0);

    let a_len = TpU8::protect(ranks_a.len() as u8);
    let b_len = TpU8::protect(ranks_b.len() as u8);

    for _ in 0..s.min(ranks_a.len() + ranks_b.len()) {
        let a = ranks_a.get(p_a.as_u32());
        let b = ranks_b.get(p_b.as_u32());

        let select_a = (p_a.tp_lt(&a_len) & p_b.tp_gt_eq(&b_len))
            | (p_a.tp_lt(&a_len) & p_b.tp_lt(&b_len) & a.tp_lt(&b));

        top_s.push(select_a.select(a, b));

        p_a = select_a.select(p_a + 1, p_a);
        p_b = select_a.select(p_b, p_b + 1);
    }
    top_s
}

#[cfg(test)]
mod tests {
    use super::*;
    use timing_shield::{TpU32, TpU64};

    #[test]
    fn select_merge_top_s() {
        let s = 4;

        let mut ranks_a = OblivVec::with_capacity(s);
        let mut ranks_b = OblivVec::with_capacity(s);

        [1, 3, 5, 7]
            .iter()
            .for_each(|&v| ranks_a.push(TpU64::protect(v)));
        [2, 4, 6, 7]
            .iter()
            .for_each(|&v| ranks_b.push(TpU64::protect(v)));

        let result = merge_top_s(s, &ranks_a, &ranks_b);
        for (i, &a) in [1, 2, 3, 4].iter().enumerate() {
            assert_eq!(result.get(TpU32::protect(i as u32)).expose(), a);
        }

        let rank = [3, 1, 5, 2, 4]
            .iter()
            .map(|&v| TpU64::protect(v))
            .collect::<Vec<_>>();

        let result = select_top_s(s, &rank);

        for (i, &a) in [1, 2, 3, 4].iter().enumerate() {
            assert_eq!(result.get(TpU32::protect(i as u32)).expose(), a);
        }
    }
}
