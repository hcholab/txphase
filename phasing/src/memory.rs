use crate::Real;

use ndarray::{ArrayView3, ArrayViewMut3};

pub struct RealMemory(Vec<Real>);

impl RealMemory {
    pub fn new() -> Self {
        Self(Vec::new())
    }
    pub fn borrow_3d(&mut self, d0: usize, d1: usize, d2: usize) -> ArrayView3<Real> {
        assert!(d0 * d1 * d2 <= self.0.len());
        ArrayView3::from_shape((d0, d1, d2), &self.0[..d0 * d1 * d2]).unwrap()
    }

    pub fn borrow_3d_mut(&mut self, d0: usize, d1: usize, d2: usize) -> ArrayViewMut3<Real> {
        if self.0.len() < d0 * d1 * d2 {
            unsafe {
                self.0
                    .resize(d0 * d1 * d2, std::mem::MaybeUninit::zeroed().assume_init());
            }
        }
        ArrayViewMut3::from_shape((d0, d1, d2), &mut self.0[..d0 * d1 * d2]).unwrap()
    }
}
