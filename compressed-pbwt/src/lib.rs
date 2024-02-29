#![feature(get_many_mut)]
#![feature(int_roundings)]
#![feature(stmt_expr_attributes)]
#![feature(is_sorted)]
#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub mod mcmc_init;
pub mod nn;
pub mod pbwt_trie;
//pub mod record;
//pub mod site;
#[cfg(not(feature = "obliv"))]
mod top_s;

#[allow(dead_code)]
pub mod test_utils;

#[cfg(feature = "obliv")]
mod inner {
    use timing_shield::*;
    pub type Bool = TpBool;
    pub type U16 = TpU16;
    pub type U32 = TpU32;
    pub type Usize = TpU64;
    pub type Isize = TpI64;
    pub type I8 = TpI8;
}

#[cfg(not(feature = "obliv"))]
mod inner {
    pub type Bool = bool;
    pub type U16 = u16;
    pub type U32 = u32;
    pub type Usize = usize;
    pub type Isize = isize;
    pub type I8 = i8;
}

pub use inner::*;
