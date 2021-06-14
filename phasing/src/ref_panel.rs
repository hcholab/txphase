use bitvec::{order::Lsb0, slice::BitSlice};
use ndarray::{Array2, ArrayViewMut2};
use std::mem::size_of;

pub struct RefPanel {
    packed: Array2<u64>,
    n_samples: usize,
    n_markers: usize,
}

impl RefPanel {
    pub fn from_vec(ref_panel: &[Vec<u8>]) -> Self {
        let n_markers = ref_panel.len();
        let n_samples = ref_panel.first().unwrap().len();
        assert!(ref_panel.iter().all(|v| v.len() == n_samples));
        let n_samples_elems = (n_samples + size_of::<u64>() - 1) / size_of::<u64>();
        let mut packed =
            unsafe { Array2::<u64>::uninit((n_markers, n_samples_elems)).assume_init() };
        for (mut packed_col, ref_col) in packed.outer_iter_mut().into_iter().zip(ref_panel.iter()) {
            let slice = packed_col.as_slice_mut().unwrap();
            let bit_slice = BitSlice::<Lsb0, u64>::from_slice_mut(slice).unwrap();
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
                    let bit_slice = BitSlice::<Lsb0, u64>::from_slice(slice).unwrap();
                    bit_slice
                        .into_iter()
                        .take(self.n_samples)
                        .map(|b| *b as u8)
                        .collect::<Vec<_>>()
                }),
        )
    }

    //pub fn iter_rows

    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    pub fn n_markers(&self) -> usize {
        self.n_markers
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn ref_panel() {
        let n_samples = 123; // # samples
        let n_markers = 45; // # markers
        let mut rng = rand::thread_rng();
        let ref_panel = crate::utils::gen_ref_panel(n_samples, n_markers, &mut rng);
        let packed_ref_panel = RefPanel::from_vec(&ref_panel);
        let result_ref_panel = packed_ref_panel.iter_columns().collect::<Vec<_>>();
        assert_eq!(ref_panel, result_ref_panel);
    }
}
