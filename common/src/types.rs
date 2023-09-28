#[cfg(feature = "obliv")]
mod inner {
    use tp_fixedpoint::timing_shield::TpBool;
    pub type Genotype = i8;
    pub type UInt = u32;
    pub type Int = i32;
    pub type U8 = u8;
    pub type Bool = bool;
    pub type BoolMcc = TpBool;
    pub type Real = f64;
    pub const F: usize = 52;
    pub type RealHmm = tp_fixedpoint::TpFixed64<F>;
}

#[cfg(not(feature = "obliv"))]
mod inner {
    pub type Genotype = i8;
    pub type UInt = u32;
    pub type Int = i32;
    pub type U8 = u8;
    pub type Bool = bool;
    pub type BoolMcc = bool;
    pub type Real = f64;
    pub type RealHmm = f64;
}

pub use inner::*;
