use crate::aligned::{merge_sort_aligned, rl_cap, Aligned};
use crate::vec::OblivVec;
use timing_shield::{TpCondSwap, TpEq, TpOrd, TpU32, TpU8};

pub fn select_top_s_merge<T>(s: usize, ranks: Vec<T>) -> OblivVec<T>
where
    [(); rl_cap::<T>()]:,
    T: TpOrd + TpCondSwap + Clone,
{
    let mut split_i = TpU32::protect(ranks.len() as u32 - 1);
    for (i, window) in ranks.windows(2).enumerate() {
        let cond = window[1].tp_gt(&window[0]);
        split_i = cond.select(TpU32::protect(i as u32), split_i);
    }

    let ranks = OblivVec::from_iter(ranks.into_iter());

    let mut ptr_up = split_i.as_i32();
    let mut ptr_down = split_i.as_i32() + 1;

    let mut item_up = ranks.get(ptr_up.as_u32());
    let mut item_down = ranks.get(ptr_down.as_u32());

    let mut some_up;
    let mut some_down;

    let mut top_s = Vec::new();
    for i in 0..s {
        some_up = ptr_up.tp_gt_eq(&0);
        some_down = ptr_down.tp_lt(&(ranks.len() as i32 - 1));

        let select_up = some_up & (!some_down | item_up.tp_lt(&item_down));
        top_s.push(select_up.select(item_up.clone(), item_down.clone()));

        if i < s {
            ptr_up = select_up.select(ptr_up - 1, ptr_up);
            ptr_down = select_up.select(ptr_down, ptr_down + 1);

            let new_item = ranks.get(select_up.select(ptr_up, ptr_down).as_u32());
            item_up = select_up.select(new_item.clone(), item_up);
            item_down = select_up.select(item_down, new_item);
        }
    }

    OblivVec::from_iter(top_s.into_iter())
}

pub fn select_top_s_stable<T>(s: usize, mut ranks: Vec<T>) -> OblivVec<T>
where
    [(); rl_cap::<T>()]:,
    T: TpOrd + TpCondSwap + Clone,
{
    if ranks.len() == 1 {
        return crate::vec::OblivVec::with_elem(1, ranks.pop().unwrap());
    }

    let mut top_s = crate::vec::OblivVec::with_elem(s, unsafe {
        std::mem::MaybeUninit::<T>::uninit().assume_init()
    });
    for j in 0..s.min(ranks.len() - 1) {
        for i in 0..ranks.len() - 1 {
            let [a, b] = unsafe { ranks.get_many_unchecked_mut([i, i + 1]) };
            a.tp_lt_eq(b).cond_swap(a, b);
        }
        top_s[j] = ranks.pop().unwrap();
    }
    top_s
}

pub fn select_top_s_2<T>(s: usize, mut ranks: Vec<T>) -> OblivVec<T>
where
    [(); rl_cap::<T>()]:,
    T: TpOrd + TpCondSwap + Clone,
{
    let (prefix, aligned, suffix) = unsafe { ranks.align_to_mut::<Aligned<T>>() };
    {
        let mut buffer = unsafe { Aligned::<T>::uninit() };
        let len = prefix.len();
        if len > 0 {
            merge_sort_aligned(prefix, &mut buffer.0[..len], true);
        }
        for aligned in aligned.iter_mut() {
            merge_sort_aligned(&mut aligned.0, &mut buffer.0, true);
        }
        let len = suffix.len();
        if len > 0 {
            merge_sort_aligned(suffix, &mut buffer.0[..len], true);
        }
    }

    let mut slices = Vec::new();
    if prefix.len() > 0 {
        slices.push((TpU8::protect(0), &*prefix));
    }
    for aligned in aligned.iter() {
        slices.push((TpU8::protect(0), &aligned.0));
    }
    if suffix.len() > 0 {
        slices.push((TpU8::protect(0), &*suffix));
    }

    let mut top_s = Vec::with_capacity(s);

    let mut max_rank: Option<T> = None;
    let mut max_slice_i = TpU32::protect(0);

    for _ in 0..s {
        for (slice_i, (i, slice)) in slices.iter_mut().enumerate() {
            let i = i.tp_lt(&(slice.len() as u8)).select(*i, TpU8::protect(0));
            let slice_item = slice[i.expose() as usize].to_owned();

            let slice_i = slice_i as u32;

            if let Some(max_rank_) = max_rank.as_mut() {
                let cond = max_rank_.tp_lt(&slice_item);
                *max_rank_ = cond.select(max_rank_.to_owned(), slice_item);
                max_slice_i = cond.select(max_slice_i, TpU32::protect(slice_i));
            } else {
                max_rank = Some(slice_item);
                max_slice_i = TpU32::protect(slice_i);
            }
        }

        top_s.push(max_rank.take().unwrap());

        for (slice_i, (i, _)) in slices.iter_mut().enumerate() {
            *i = max_slice_i.tp_eq(&(slice_i as u32)).select(*i + 1, *i);
        }

        max_slice_i = TpU32::protect(0);
    }

    crate::vec::OblivVec::from_iter(top_s.into_iter())
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

        let result = select_top_s(s, rank);

        for (i, &a) in [1, 2, 3, 4].iter().enumerate() {
            assert_eq!(result.get(TpU32::protect(i as u32)).expose(), a);
        }

        let rank = (1..100)
            .rev()
            .map(|v| TpU64::protect(v))
            .collect::<Vec<_>>();

        let result = select_top_s_2(s, rank);
        for (i, &a) in [1, 2, 3, 4].iter().enumerate() {
            assert_eq!(result.get(TpU32::protect(i as u32)).expose(), a);
        }
    }
}
