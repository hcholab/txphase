use bitvec::{order::Lsb0, slice::BitSlice};
use ndarray::{Array2, ArrayViewMut2};
use std::mem::size_of;

pub struct RefPanel {
    packed: Array2<usize>,
    n_samples: usize,
    n_markers: usize,
}

impl RefPanel {
    pub fn from_vec(ref_panel: &[Vec<u8>]) -> Self {
        let n_markers = ref_panel.len();
        let n_samples = ref_panel.first().unwrap().len();
        assert!(ref_panel.iter().all(|v| v.len() == n_samples));
        let n_samples_elems = (n_samples + size_of::<usize>() - 1) / size_of::<usize>();
        let mut packed =
            unsafe { Array2::<usize>::uninit((n_markers, n_samples_elems)).assume_init() };
        for (mut packed_col, ref_col) in packed.outer_iter_mut().into_iter().zip(ref_panel.iter()) {
            let slice = packed_col.as_slice_mut().unwrap();
            let bit_slice = BitSlice::<Lsb0>::from_slice_mut(slice).unwrap();
            for (mut tar_bit, &src_bit) in bit_slice.into_iter().zip(ref_col.iter()) {
                *tar_bit = src_bit != 0;
            }
        }
        Self {
            n_samples,
            n_markers,
            packed,
        }
    }

    pub fn iter_columns<'a>(&'a self) -> Box<dyn Iterator<Item = Vec<u8>> + 'a> {
        Box::new(
            self.packed
                .outer_iter()
                .into_iter()
                .take(self.n_markers)
                .map(move |column| {
                    let slice = column.as_slice().unwrap();
                    let bit_slice = BitSlice::<Lsb0>::from_slice(slice).unwrap();
                    bit_slice
                        .into_iter()
                        .take(self.n_samples)
                        .map(|b| *b as u8)
                        .collect::<Vec<_>>()
                }),
        )
    }
}
