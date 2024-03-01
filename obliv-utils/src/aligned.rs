use timing_shield::{TpBool, TpCondSwap, TpOrd, TpU8};

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
            let [a, b] = self.0.get_many_mut([i - 1, i]).unwrap();
            let do_swap = b.tp_lt(&a);
            do_swap.cond_swap(a, b);
        }
    }

    pub fn obliv_bubble_sort_last(&mut self) {
        self.obliv_bubble_sort_pos(self.0.len() - 1);
    }
}

//impl<T> Aligned<T>
//where
//[(); rl_cap::<T>()]:,
//T: TpOrd + TpCondSwap + Clone,
//{
//pub fn obliv_merge_sort_pos(&mut self, pos: usize) {
//let mut buffer = Self::default();
//merge_sort_aligned(&mut self.0[..pos + 1], &mut buffer.0[..pos + 1])
//}

//pub fn obliv_merge_sort_last(&mut self) {
//let mut buffer = Self::default();
//merge_sort_aligned(&mut self.0, &mut buffer.0)
//}
//}

pub fn merge_sort_aligned<T: TpOrd + TpCondSwap + Clone>(
    items: &mut [T],
    buffer: &mut [T],
    ascending: bool,
) {
    assert_eq!(items.len(), buffer.len());
    if items.len() == 2 {
        let [a, b] = unsafe { items.get_many_unchecked_mut([0, 1]) };
        (!(ascending ^ a.tp_gt(b))).cond_swap(a, b);
    } else if items.len() > 2 {
        {
            let (items_a, items_b) = items.split_at_mut(items.len() / 2);
            let (buffer_a, buffer_b) = buffer.split_at_mut(buffer.len() / 2);
            merge_sort_aligned(items_a, buffer_a, ascending);
            merge_sort_aligned(items_b, buffer_b, ascending);
        }

        buffer.clone_from_slice(items);
        let (buffer_a, buffer_b) = buffer.split_at(items.len() / 2);

        let mut a_ptr = TpU8::protect(0);
        let mut b_ptr = TpU8::protect(0);
        let (mut some_a, mut cur_a) = (TpBool::protect(true), &buffer_a[0]);
        let (mut some_b, mut cur_b) = (TpBool::protect(true), &buffer_b[0]);

        for item in items.iter_mut() {
            let select_a = some_a & ((!some_b) | (ascending ^ cur_a.tp_gt(cur_b)));
            *item = select_a.select(cur_a.to_owned(), cur_b.to_owned());
            a_ptr = select_a.select(a_ptr + 1, a_ptr);
            b_ptr = select_a.select(b_ptr, b_ptr + 1);
            (some_a, cur_a) = get_item(&buffer_a, a_ptr);
            (some_b, cur_b) = get_item(&buffer_b, b_ptr);
        }
    }
}

fn get_item<T: Clone>(items: &[T], index: TpU8) -> (TpBool, &T) {
    let cond = index.tp_lt(&(items.len() as u8));
    let index = cond.select(index, TpU8::protect(0));
    (cond, &items[index.expose() as usize])
}
