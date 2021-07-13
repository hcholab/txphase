#![allow(dead_code)]
#![feature(destructuring_assignment)]
mod neighbors_finding;
mod pbwt;
//mod initialize;
mod union_filter;
//mod ref_panel;
mod genotype_graph;
mod hmm;
mod hmm2;
mod utils;

#[cfg(feature = "leak-resist")]
mod inner {
    use tp_fixedpoint::timing_shield::{TpBool, TpI8, TpU32, TpU8};
    use tp_fixedpoint::TpLnFixed;
    pub type Genotype = TpI8;
    pub type UInt = TpU32;
    pub type U8 = TpU8;
    pub type Bool = TpBool;
    pub type Real = TpLnFixed<20>;
}

#[cfg(not(feature = "leak-resist"))]
mod inner {
    pub type Genotype = i8;
    pub type UInt = u32;
    pub type U8 = u8;
    pub type Bool = bool;
    pub type Real = f32;
}

use inner::*;


fn main() {
    hmm2::hmm();
}
