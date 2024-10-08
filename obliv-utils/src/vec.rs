use crate::aligned::{rl_cap, Aligned};
use timing_shield::{TpBool, TpCondSwap, TpEq, TpOrd, TpU32, TpU8};

#[derive(Clone)]
pub struct OblivVec<T>
where
    [(); rl_cap::<T>()]:,
{
    inner: Vec<Aligned<T>>,
    len: u32,
    last_len: u8,
}

impl<T> OblivVec<T>
where
    [(); rl_cap::<T>()]:,
{
    pub fn new() -> Self {
        Self {
            inner: unsafe { vec![Aligned::<T>::uninit()] },
            len: 0,
            last_len: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let mut inner = Vec::with_capacity(capacity.div_ceil(rl_cap::<T>()));
        inner.push(unsafe { Aligned::<T>::uninit() });
        Self {
            inner,
            len: 0,
            last_len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn push(&mut self, item: T) {
        if self.last_len as usize == rl_cap::<T>() {
            self.last_len = 0;
            self.inner.push(unsafe { Aligned::<T>::uninit() });
        }
        self.inner.last_mut().unwrap().0[self.last_len as usize] = item;
        self.last_len += 1;
        self.len += 1;
    }
    pub fn first(&self) -> Option<&T> {
        self.inner.first().unwrap().0.first()
    }
    pub fn last(&self) -> Option<&T> {
        self.inner.last().map(|v| &v.0[self.last_len as usize - 1])
    }

    pub fn last_mut(&mut self) -> Option<&mut T> {
        self.inner
            .last_mut()
            .map(|v| &mut v.0[self.last_len as usize - 1])
    }

    pub fn into_iter(self) -> impl Iterator<Item = T> {
        let len = self.len();
        self.inner
            .into_iter()
            .map(|v| v.0.into_iter())
            .flatten()
            .take(len)
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> {
        let len = self.len();
        self.inner.iter().map(|v| v.0.iter()).flatten().take(len)
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        let len = self.len();
        self.inner
            .iter_mut()
            .map(|v| v.0.iter_mut())
            .flatten()
            .take(len)
    }
    pub fn cond_copy_from(&mut self, src: &Self, cond: TpBool) {
        assert_eq!(self.len(), src.len());

        let capacity = rl_cap::<T>();
        #[cfg(target_feature = "avx2")]
        {
            use crate::cmov::cmov_byte_slice_a64;
            let num_bytes = self.len().div_ceil(capacity) * 64;
            unsafe {
                cmov_byte_slice_a64(
                    cond.expose(),
                    src.inner.as_ptr() as *const u64,
                    self.inner.as_mut_ptr() as *mut u64,
                    num_bytes as usize,
                );
            }
        }
        #[cfg(not(target_feature = "avx2"))]
        {
            use crate::cmov::cmov_byte_slice_a8;
            let count = self.len().div_ceil(capacity) * 64 / 8;
            unsafe {
                cmov_byte_slice_a8(
                    cond.expose(),
                    src.inner.as_ptr() as *const u64,
                    self.inner.as_mut_ptr() as *mut u64,
                    count as usize,
                );
            }
        }
    }

    fn cal_ind(&self, i: TpU32) -> (TpU32, TpU8) {
        match log2::<T>() {
            Some(n) => (i >> n, (i & ((1 << n) - 1)).as_u8()),
            None => {
                let i = i
                    .tp_gt_eq(&(self.len() as u32))
                    .select(TpU32::protect(0), i);
                let mut aligned_ind = TpU32::protect(0);
                let mut inner_ind = TpU8::protect(0);
                let aligned_len = rl_cap::<T>() as u32;
                let mut cur = TpU32::protect(0);
                for j in 0..self.inner.len() as u32 {
                    let cond = i.tp_gt_eq(&cur) & i.tp_lt(&(cur + aligned_len));
                    aligned_ind = cond.select(TpU32::protect(j), aligned_ind);
                    inner_ind = cond.select((i - cur).as_u8(), inner_ind);
                    cur += aligned_len;
                }
                (aligned_ind, inner_ind)
            }
        }
    }
}

impl<T> OblivVec<T>
where
    [(); rl_cap::<T>()]:,
    T: Clone,
{
    pub fn with_elem(n: usize, elem: T) -> Self {
        let aligned = Aligned::with_elem(elem);
        let inner = vec![aligned; n.div_ceil(rl_cap::<T>())];
        Self {
            inner,
            len: n as u32,
            last_len: (n % rl_cap::<T>()) as u8,
        }
    }
}

impl<T> OblivVec<T>
where
    [(); rl_cap::<T>()]:,
    T: Default + Clone,
{
    pub fn default(len: usize) -> Self {
        let n_aligned = len.div_ceil(rl_cap::<T>());
        Self {
            inner: vec![Aligned::<T>::default(); n_aligned],
            len: len as u32,
            last_len: (len % rl_cap::<T>()) as u8,
        }
    }
}

impl<T> OblivVec<T>
where
    [(); rl_cap::<T>()]:,
    T: Clone + TpCondSwap,
{
    pub fn get(&self, i: TpU32) -> T {
        let mut item = unsafe { std::mem::MaybeUninit::<T>::uninit().assume_init() };
        let (alinged_ind, inner_ind) = self.cal_ind(i);
        for (i, aligned) in self.inner.iter().enumerate() {
            item = alinged_ind
                .tp_eq(&(i as u32))
                .select(aligned.0[inner_ind.expose() as usize].clone(), item);
        }
        item
    }

    pub fn get_many(&self, indices: &[TpU32]) -> Vec<T> {
        let mut items =
            unsafe { vec![std::mem::MaybeUninit::<T>::uninit().assume_init(); indices.len()] };
        let indices = indices.iter().map(|&i| self.cal_ind(i)).collect::<Vec<_>>();
        for (i, aligned) in self.inner.iter().enumerate() {
            for (&(aligned_ind, inner_ind), item) in indices.iter().zip(&mut items) {
                *item = aligned_ind
                    .tp_eq(&(i as u32))
                    .select(aligned.0[inner_ind.expose() as usize].clone(), item.clone());
            }
        }
        items
    }

    // Return original value
    pub fn set(&mut self, i: TpU32, mut item: T) -> T {
        let (aligned_ind, inner_ind) = self.cal_ind(i);
        for (i, aligned) in self.inner.iter_mut().enumerate() {
            let target = &mut aligned.0[inner_ind.expose() as usize];
            aligned_ind.tp_eq(&(i as u32)).cond_swap(target, &mut item);
        }
        item
    }

    pub fn apply(&mut self, i: TpU32, mut f: impl FnMut(&mut T)) {
        let (alinged_ind, inner_ind) = self.cal_ind(i);
        for (i, aligned) in self.inner.iter_mut().enumerate() {
            let target = &mut aligned.0[inner_ind.expose() as usize];
            let mut target_mut = target.clone();
            f(&mut target_mut);
            alinged_ind
                .tp_eq(&(i as u32))
                .cond_swap(target, &mut target_mut);
        }
    }

    pub fn cond_copy_from_slice(&mut self, src: &[T], cond: TpBool) {
        assert_eq!(self.len(), src.len());
        for (target, source) in self.iter_mut().zip(src.iter()) {
            *target = cond.select(source.clone(), target.clone());
        }
    }
}

impl<T> OblivVec<T>
where
    [(); rl_cap::<T>()]:,
    T: TpOrd + TpCondSwap,
{
    pub(crate) fn bubble_sort_last(&mut self) {
        for i in (1..self.inner.len()).rev() {
            let len = self.inner.len();
            let [a, b] = self.inner.get_many_mut([i - 1, i]).unwrap();
            if i == len - 1 {
                b.obliv_bubble_sort_pos(self.last_len as usize - 1);
            } else {
                b.obliv_bubble_sort_last();
            }
            let b_first = &mut b.0[0];
            let a_last = a.0.last_mut().unwrap();

            let do_swap = b_first.tp_lt(&a_last);
            do_swap.cond_swap(a_last, b_first);
        }

        if self.inner.len() == 1 {
            self.inner[0].obliv_bubble_sort_pos(self.last_len as usize - 1);
        } else {
            self.inner[0].obliv_bubble_sort_last();
        }
    }
}

impl<T> FromIterator<T> for OblivVec<T>
where
    [(); rl_cap::<T>()]:,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut rank_list = OblivVec::<T>::new();
        for i in iter {
            rank_list.push(i);
        }

        rank_list
    }
}

impl<T> std::ops::IndexMut<usize> for OblivVec<T>
where
    [(); rl_cap::<T>()]:,
{
    fn index_mut(&mut self, index: usize) -> &mut T {
        if index >= self.len() {
            panic!();
        }
        &mut self.inner[index / rl_cap::<T>()].0[index % rl_cap::<T>()]
    }
}

//impl<T> std::ops::Index<usize> for OblivVec<T>
//where
//[(); rl_cap::<T>()]:,
//{
//fn index(&self, index: usize) -> &T {
//&mut self.inner[index/rl_cap::<T>()].0[index%rl_cap::<T>()]
//}
//}

impl<T> Extend<T> for OblivVec<T>
where
    [(); rl_cap::<T>()]:,
{
    fn extend<S>(&mut self, iter: S)
    where
        S: IntoIterator<Item = T>,
    {
        for item in iter {
            self.push(item);
        }
    }
}

impl<T> std::ops::Index<usize> for OblivVec<T>
where
    [(); rl_cap::<T>()]:,
{
    type Output = T;
    fn index(&self, index: usize) -> &T {
        if index >= self.len() {
            panic!();
        }
        let capacity = rl_cap::<T>();
        &self.inner[index / capacity].0[index % capacity]
    }
}

const fn log2<T>() -> Option<u32> {
    if rl_cap::<T>().is_power_of_two() {
        Some(rl_cap::<T>().ilog2())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obliv_vec() {
        let ref_numbers = (0..2000u32).collect::<Vec<_>>();
        let ref_numbers_tp = ref_numbers
            .iter()
            .map(|v| timing_shield::TpU32::protect(*v))
            .collect::<Vec<_>>();
        let mut vec = OblivVec::with_capacity(ref_numbers_tp.len());
        for i in ref_numbers_tp {
            vec.push(i);
        }
        let test_numbers = vec.iter().map(|v| v.expose()).collect::<Vec<_>>();

        assert_eq!(ref_numbers, test_numbers);

        for &i in &ref_numbers {
            assert_eq!(vec.get(TpU32::protect(i as u32)).expose(), i);
        }

        for i in 0..2000u32 {
            let orig = vec.set(TpU32::protect(i), TpU32::protect(i + 1));
            assert_eq!(orig.expose(), i);
        }

        let test_numbers = vec.iter().map(|v| v.expose()).collect::<Vec<_>>();

        assert_eq!(test_numbers, (1..2001).collect::<Vec<_>>());
    }
}
