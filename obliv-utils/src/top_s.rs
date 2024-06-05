use crate::aligned::rl_cap;
use crate::vec::OblivVec;
use timing_shield::{TpCondSwap, TpOrd, TpU8};

pub fn select_top_s_stable<T>(s: usize, mut ranks: Vec<T>) -> OblivVec<T>
where
    [(); rl_cap::<T>()]:,
    T: TpOrd + TpCondSwap + Clone,
{
    if ranks.len() == 1 {
        return crate::vec::OblivVec::with_elem(1, ranks.pop().unwrap());
    }

    let mut top_s = crate::vec::OblivVec::with_elem(s.min(ranks.len()), unsafe {
        std::mem::MaybeUninit::<T>::uninit().assume_init()
    });
    for j in 0..s.min(ranks.len()) {
        for i in 0..ranks.len() - 1 {
            let [a, b] = unsafe { ranks.get_many_unchecked_mut([i, i + 1]) };
            a.tp_lt_eq(b).cond_swap(a, b);
        }
        top_s[j] = ranks.pop().unwrap();
    }
    top_s
}

pub fn select_top_s<T>(s: usize, mut ranks: Vec<T>) -> OblivVec<T>
where
    [(); rl_cap::<T>()]:,
    T: TpOrd + TpCondSwap + Clone,
{
    if ranks.len() == 1 {
        return crate::vec::OblivVec::with_elem(1, ranks.pop().unwrap());
    }

    if ranks.len() <= s {
        crate::bitonic_sort::bitonic_sort(&mut ranks, true);
        return ranks.into_iter().collect();
    }

    let mut top_s = crate::vec::OblivVec::with_elem(s, unsafe {
        std::mem::MaybeUninit::<T>::uninit().assume_init()
    });
    for j in 0..s {
        for i in 0..ranks.len() - 1 {
            let [a, b] = unsafe { ranks.get_many_unchecked_mut([i, i + 1]) };
            a.tp_lt(b).cond_swap(a, b);
        }
        top_s[j] = ranks.pop().unwrap();
    }
    top_s
}

// assume the ranks are sorted
pub fn merge_top_s<T>(s: usize, ranks_a: &OblivVec<T>, ranks_b: &OblivVec<T>) -> OblivVec<T>
where
    [(); rl_cap::<T>()]:,
    T: Clone + TpOrd + TpCondSwap,
{
    let mut top_s = crate::vec::OblivVec::with_elem(s.min(ranks_a.len() + ranks_b.len()), unsafe {
        std::mem::MaybeUninit::<T>::uninit().assume_init()
    });

    let mut p_a = TpU8::protect(0);
    let mut p_b = TpU8::protect(0);

    let a_len = TpU8::protect(ranks_a.len() as u8);
    let b_len = TpU8::protect(ranks_b.len() as u8);

    for i in 0..s.min(ranks_a.len() + ranks_b.len()) {
        let mut a = ranks_a.get(p_a.as_u32());
        let mut b = ranks_b.get(p_b.as_u32());

        let select_a = (p_a.tp_lt(&a_len) & p_b.tp_gt_eq(&b_len))
            | (p_a.tp_lt(&a_len) & p_b.tp_lt(&b_len) & a.tp_lt(&b));

        (!select_a).cond_swap(&mut b, &mut a);

        top_s[i] = a;

        p_a = select_a.select(p_a + 1, p_a);
        p_b = select_a.select(p_b, p_b + 1);
    }
    top_s
}

#[cfg(test)]
mod tests {
    use super::*;
    use timing_shield::{TpBool, TpU32, TpU64};

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

        let result = select_top_s(s, rank);

        for (i, &a) in [1, 2, 3, 4].iter().enumerate() {
            assert_eq!(result.get(TpU32::protect(i as u32)).expose(), a);
        }

        let mut rank = [
            (5, 2),
            (5, 1),
            (1, 2),
            (1, 1),
            (3, 2),
            (3, 1),
            (4, 1),
            (4, 2),
            (2, 1),
            (2, 2),
        ];

        let obliv_rank = rank
            .iter()
            .map(|v| ToSort {
                order: TpU64::protect(v.0),
                payload: TpU64::protect(v.1),
            })
            .collect::<Vec<_>>();

        let obliv_rank = select_top_s_stable(8, obliv_rank);
        let obliv_rank = obliv_rank
            .into_iter()
            .map(|v| (v.order.expose(), v.payload.expose()))
            .collect::<Vec<_>>();

        rank.sort_by_key(|v| v.0);
        let rank = rank.into_iter().take(8).collect::<Vec<_>>();
        assert_eq!(obliv_rank, rank);
    }

    #[derive(Clone)]
    struct ToSort {
        order: TpU64,
        payload: TpU64,
    }

    impl timing_shield::TpCondSwap for ToSort {
        fn tp_cond_swap(cond: TpBool, a: &mut Self, b: &mut Self) {
            cond.cond_swap(&mut a.order, &mut b.order);
            cond.cond_swap(&mut a.payload, &mut b.payload);
        }
    }

    impl timing_shield::TpOrd for ToSort {
        fn tp_lt(&self, _: &ToSort) -> TpBool {
            todo!()
        }
        fn tp_lt_eq(&self, other: &ToSort) -> TpBool {
            self.order.tp_lt_eq(&other.order)
        }
        fn tp_gt(&self, _: &ToSort) -> TpBool {
            todo!()
        }
        fn tp_gt_eq(&self, _: &ToSort) -> TpBool {
            todo!()
        }
    }
}
