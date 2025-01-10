#![feature(generic_const_exprs)]
#![feature(get_many_mut)]
#![allow(incomplete_features)]
#![feature(pointer_is_aligned_to)]
#![feature(int_roundings)]
#![feature(slice_take)]
#![feature(portable_simd)]
#![feature(stdarch_x86_avx512)]
#![feature(const_slice_from_raw_parts_mut)]

pub mod aligned;
pub mod bitmap;
pub mod bitonic_sort;
pub mod cmov;
pub mod cond_copy;
pub mod filter;
pub mod min_heap;
pub mod nt_store;
pub mod top_s;
pub mod utils;
pub mod vec;
