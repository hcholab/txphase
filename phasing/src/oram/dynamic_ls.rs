use maligned::{align_first, A64};
use ndarray::ArrayView2;
use std::mem::size_of;
use timing_shield::{TpBool, TpEq, TpU32};

fn new_empty_aligned_vec<T: Clone>(n_elems: usize) -> Vec<T> {
    let mut v = align_first::<T, A64>(n_elems);
    v.resize(n_elems, unsafe {
        std::mem::MaybeUninit::uninit().assume_init()
    });
    v
}

fn clone_aligned_slice<T: Clone>(src: &[T]) -> Vec<T> {
    let mut new_v = new_empty_aligned_vec(src.len());
    new_v.clone_from_slice(src);
    new_v
}

pub struct DynamicLSOram<T: Clone> {
    inner: Vec<Vec<T>>,
    n_elems_per_row: usize,
    row_size_bytes_rounded: usize,
    n_elems_per_row_with_padding: usize,
}

impl<T: Clone> DynamicLSOram<T> {
    pub fn new(n_rows: usize, n_elems_per_row: usize) -> Self {
        let row_size_bytes_rounded = ((n_elems_per_row * size_of::<T>() + 63) / 64) * 64;
        let n_elems_per_row_with_padding =
            (row_size_bytes_rounded + size_of::<T>() - 1) / size_of::<T>();
        let inner = (0..n_rows)
            .map(|_| new_empty_aligned_vec(n_elems_per_row_with_padding))
            .collect();
        Self {
            inner,
            n_elems_per_row,
            row_size_bytes_rounded,
            n_elems_per_row_with_padding,
        }
    }

    pub fn from_array(array: ArrayView2<T>) -> Self {
        let mut new_self = Self::new(array.nrows(), array.ncols());
        for (dest, src) in new_self.inner.iter_mut().zip(array.rows().into_iter()) {
            (&mut dest[..src.len()]).clone_from_slice(src.as_slice().unwrap());
        }
        new_self
    }

    pub fn cond_obliv_read(&self, index: TpU32, cond: TpBool) -> Vec<T> {
        let mut out = new_empty_aligned_vec(self.n_elems_per_row_with_padding);
        for (i, row) in self.inner.iter().enumerate() {
            let is_target_row = index.tp_eq(&(i as u32)) & cond;
            cmov(
                row.as_ptr(),
                out.as_mut_ptr(),
                self.row_size_bytes_rounded,
                is_target_row,
            );
        }
        out.truncate(self.n_elems_per_row);
        out
    }

    pub fn obliv_read(&self, index: TpU32) -> Vec<T> {
        self.cond_obliv_read(index, TpBool::protect(true))
    }

    pub fn obliv_filter(&self, filter: &[TpBool], capacity: usize) -> (Vec<Vec<T>>, TpU32) {
        assert_eq!(filter.len(), self.inner.len());
        //let mut filtered = vec![Vec::new(); capacity];
        for (filter_chunk, data_chunk) in filter.chunks(capacity).zip(self.inner.chunks(capacity)) {
        }
        todo!()
    }

    //fn to_obliv_items<T>(
        //bitmap: &[TpBool],
        //all_data: &[Vec<T>],
        //true_start: TpU32,
        //resize: usize,
    //) -> Vec<OblivFilterItem<T>> {
        //assert_eq!(bitmap.len(), all_data.len());
        //assert!(resize >= all_data.len());
        //let nummap = if all_data.len() < resize {
            //let mut bitmap = bitmap.to_vec();
            //bitmap.resize(resize, TpBool::protect(false));
            //assign_bitmap(&bitmap, true_start)
        //} else {
            //assign_bitmap(bitmap, true_start)
        //};
        //let blank;
        //let data_iter: Box<dyn Iterator<Item = &Vec<T>>> = if all_data.len() < resize {
            //blank = Vec::with_capacity(0);
            //Box::new(all_data.iter().chain(std::iter::repeat(&blank)))
        //} else {
            //Box::new(all_data.iter())
        //};
        //nummap
            //.into_iter()
            //.zip(data_iter)
            //.map(|(num, data)| OblivFilterItem {
                //num,
                //data: Box::new(data.clone()),
            //})
        //.collect()
    //}
}

use timing_shield::{TpCondSwap, TpOrd, TpU8};


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

struct OblivFilterItem<T> {
    num: TpU8,
    data: Vec<T>,
}

impl<T> OblivFilterItem<T> {
    pub fn split(self) -> (TpBool, Vec<T>) {
        ((self.num & 1).tp_not_eq(&0), self.data)
    }
}

impl<T> TpOrd for OblivFilterItem<T> {
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

impl<T: Clone> TpCondSwap for OblivFilterItem<T> {
    fn tp_cond_swap(condition: TpBool, a: &mut Self, b: &mut Self) {
        condition.cond_swap(&mut a.num, &mut b.num);
        let mut temp = align_first::<T, A64>(a.data.len());
        temp.resize(a.data.len(), unsafe {
            std::mem::MaybeUninit::uninit().assume_init()
        });
        cmov(
            a.data.as_ptr(),
            temp.as_mut_ptr(),
            size_of::<T>() * a.data.len(),
            condition,
        );
        cmov(
            b.data.as_ptr(),
            a.data.as_mut_ptr(),
            size_of::<T>() * a.data.len(),
            condition,
        );
        cmov(
            temp.as_ptr(),
            b.data.as_mut_ptr(),
            size_of::<T>() * a.data.len(),
            condition,
        );
    }
}

fn cmov<T>(src: *const T, dest: *mut T, n_bytes: usize, cond: TpBool) {
    let src = src as *const u64;
    let dest = dest as *mut u64;
    unsafe {
        #[cfg(target_feature = "avx2")]
        {
            assert_eq!(n_bytes % 64, 0);
            super::align::cmov_byte_slice_a64(cond.expose(), src, dest, n_bytes);
        }

        #[cfg(not(target_feature = "avx2"))]
        {
            assert_eq!(n_bytes % 8, 0);
            super::align::cmov_byte_slice_a8(cond.expose(), src, dest, n_bytes / 8);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn dynamic_ls_oram() {
        let ref_data = ndarray::Array2::<u64>::from_shape_fn((10, 20), |(i, j)| (i * j) as u64);
        let oram = DynamicLSOram::from_array(ref_data.view());

        let index: usize = 2;
        let res = oram.obliv_read(TpU32::protect(index as u32));
        assert_eq!(ref_data.row(index).as_slice().unwrap(), res.as_slice());
    }
}
