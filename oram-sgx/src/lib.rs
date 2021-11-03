#![feature(llvm_asm)]
#![feature(asm)]
mod backend;
mod bitonic_sort;
mod obliv_filter;
mod obliv_vec;
mod small_oram;
pub mod align;
pub mod utils;

pub use backend::*;
pub use bitonic_sort::*;
pub use obliv_filter::*;
pub use small_oram::*;
pub use timing_shield;

#[macro_export]
macro_rules! oram_value {
    ($struct_name: ident) => {
        impl $struct_name {
            pub unsafe fn as_slice(&self) -> &[u8] {
                let ptr = self as *const _ as *const u8;
                std::slice::from_raw_parts(ptr, std::mem::size_of::<Self>())
            }

            pub unsafe fn from_slice(slice: &[u8]) -> Self {
                assert_eq!(slice.len(), std::mem::size_of::<Self>());
                let mut uninit_self: Self = std::mem::MaybeUninit::uninit().assume_init();
                (&mut uninit_self as *mut _ as *mut u8).copy_from(slice.as_ptr(), slice.len());
                uninit_self
            }
        }
    };
}
