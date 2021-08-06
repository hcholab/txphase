use crate::{UInt, Bool}; 
#[cfg(feature = "leak-resist")]
use tp_fixedpoint::timing_shield::TpOrd;

pub struct SmallORAM<T: 'static + Clone> {
    slice_inner: &'static mut [T],
    inner: Box<oram_sgx::align::A64Bytes<64>>,
    capacity: UInt,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: 'static + Clone> SmallORAM<T> {
    pub fn with_capacity(capacity: usize) -> Self {
        assert!((capacity + 1) <= 64 / std::mem::size_of::<T>());
        let mut inner = Box::new(oram_sgx::align::A64Bytes::default());
        let inner_ptr = inner.as_mut_slice() as *mut _ as *mut T;
        let slice_inner = unsafe { std::slice::from_raw_parts_mut(inner_ptr, capacity + 1) };

        let capacity = capacity as u32;
        #[cfg(feature = "leak-resist")]
        let capacity = UInt::protect(capacity);

        Self {
            slice_inner,
            inner,
            capacity,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn write(&mut self, v: T, i: UInt, do_write: Bool) {
        #[cfg(feature = "leak-resist")]
        let i = (i.tp_gt_eq(&self.capacity) | !do_write).select(self.capacity, i).expose();

        #[cfg(not(feature = "leak-resist"))]
        {
            assert!(i < self.capacity);
            if !do_write {
                return;
            }
        }

        self.slice_inner[i as usize] = v;
    }

    pub fn read(&self, i: UInt, do_read: Bool) -> T {
        #[cfg(feature = "leak-resist")]
        let i = (i.tp_gt(&self.capacity) | !do_read).select(self.capacity, i).expose();

        #[cfg(not(feature = "leak-resist"))]
        {
            assert!(i < self.capacity);
            if !do_read{
                return self.slice_inner[0].clone();
            }
        }

        self.slice_inner[i as usize].clone()
    }
}
