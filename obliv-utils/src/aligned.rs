use timing_shield::{TpBool, TpCondSwap, TpOrd};

pub const fn rl_cap<T>() -> usize {
    64 / std::mem::size_of::<T>()
}

#[repr(C, align(64))]
#[derive(Clone)]
pub struct Aligned<T>(pub [T; rl_cap::<T>()])
where
    [(); rl_cap::<T>()]:;

impl<T> Aligned<T>
where
    [(); rl_cap::<T>()]:,
{
    pub unsafe fn uninit() -> Self {
        Self(std::mem::MaybeUninit::uninit().assume_init())
    }
}

impl<T> Aligned<T>
where
    [(); rl_cap::<T>()]:,
    T: Clone,
{
    pub fn with_elem(elem: T) -> Self {
        let mut new_self = Self(unsafe { std::mem::MaybeUninit::uninit().assume_init() });
        new_self.0.fill(elem);
        new_self
    }
}

impl<T> Default for Aligned<T>
where
    [(); rl_cap::<T>()]:,
    T: Default + Clone,
{
    fn default() -> Self {
        let mut new_self = Self(unsafe { std::mem::MaybeUninit::uninit().assume_init() });
        new_self.0.fill(T::default());
        new_self
    }
}

impl<T> Aligned<T>
where
    [(); rl_cap::<T>()]:,
    T: TpOrd + TpCondSwap,
{
    pub fn obliv_bubble_sort_pos(&mut self, pos: usize) {
        for i in (1..pos + 1).rev() {
            let [a, b] = self.0.get_disjoint_mut([i - 1, i]).unwrap();
            let do_swap = b.tp_lt(&a);
            do_swap.cond_swap(a, b);
        }
    }

    pub fn obliv_bubble_sort_last(&mut self) {
        self.obliv_bubble_sort_pos(self.0.len() - 1);
    }
}

impl<T> TpCondSwap for Aligned<T>
where
    [(); rl_cap::<T>()]:,
{
    fn tp_cond_swap(cond: TpBool, a: &mut Self, b: &mut Self) {
        use tp_fixedpoint::TpU64x8;
        unsafe {
            let a = &mut *(a as *mut _ as *mut TpU64x8);
            let b = &mut *(b as *mut _ as *mut TpU64x8);
            TpU64x8::tp_cond_swap(cond, a, b);
        }
    }
}
