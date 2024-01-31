use derive_more::From;
use timing_shield::TpI64;

#[derive(From, Clone, Copy)]
pub struct TpI128(i128);

impl TpI128 {
    pub const ZERO: Self = Self(0);

    pub fn dot(u: &[Self], v: &[Self]) -> Self {
        assert_eq!(u.len(), v.len());
        let mut xs = u;
        let mut ys = v;

        let mut s = Self::ZERO;
        let (mut p0, mut p1, mut p2, mut p3, mut p4, mut p5, mut p6, mut p7) = (
            Self::ZERO,
            Self::ZERO,
            Self::ZERO,
            Self::ZERO,
            Self::ZERO,
            Self::ZERO,
            Self::ZERO,
            Self::ZERO,
        );

        while xs.len() >= 8 {
            p0 = p0 + xs[0] * ys[0];
            p1 = p1 + xs[1] * ys[1];
            p2 = p2 + xs[2] * ys[2];
            p3 = p3 + xs[3] * ys[3];
            p4 = p4 + xs[4] * ys[4];
            p5 = p5 + xs[5] * ys[5];
            p6 = p6 + xs[6] * ys[6];
            p7 = p7 + xs[7] * ys[7];

            xs = &xs[8..];
            ys = &ys[8..];
        }
        s = s + p0 + p4;
        s = s + p1 + p5;
        s = s + p2 + p6;
        s = s + p3 + p7;

        for i in 0..xs.len() {
            s = s + xs[i] * ys[i];
        }
        s
    }
}

impl TpI128 {
    pub fn protect(v: i128) -> Self {
        Self(v)
    }
}

impl From<TpI64> for TpI128 {
    fn from(v: TpI64) -> Self {
        (v.expose() as i128).into()
    }
}

impl Into<TpI64> for TpI128 {
    fn into(self) -> TpI64 {
        TpI64::protect(self.0 as i64)
    }
}

impl std::ops::Add<Self> for TpI128 {
    type Output = Self;
    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(self.0.wrapping_add(rhs.0))
    }
}

impl std::ops::Sub<Self> for TpI128 {
    type Output = Self;
    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(self.0.wrapping_sub(rhs.0))
    }
}

impl std::ops::Mul<Self> for TpI128 {
    type Output = Self;
    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        Self(self.0.wrapping_mul(rhs.0))
    }
}

impl std::ops::Shr<u32> for TpI128 {
    type Output = Self;
    #[inline]
    fn shr(self, rhs: u32) -> Self::Output {
        Self(self.0.wrapping_shr(rhs))
    }
}

impl num_traits::One for TpI128 {
    fn one() -> Self {
        Self(1)
    }
}
