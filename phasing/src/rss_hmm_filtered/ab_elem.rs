use crate::Real;
use tp_fixedpoint::timing_shield::{TpBool, TpCondSwap, TpOrd, TpU16};
use tp_fixedpoint::TpU64x8;

#[derive(Clone)]
#[allow(invalid_value)]
pub struct AbElem {
    pub id: TpU16,
    pub e11: [TpU64x8; 8],
    pub e10: TpU64x8,
    pub e01: TpU64x8,
    pub e00: Real,
}

impl AbElem {
    pub unsafe fn uninit() -> Self {
        std::mem::MaybeUninit::zeroed().assume_init()
    }
}

impl TpOrd for AbElem {
    #[inline(always)]
    fn tp_lt(&self, other: &AbElem) -> TpBool {
        self.id.tp_lt(&other.id)
    }

    #[inline(always)]
    fn tp_lt_eq(&self, other: &AbElem) -> TpBool {
        self.id.tp_lt_eq(&other.id)
    }

    #[inline(always)]
    fn tp_gt(&self, other: &AbElem) -> TpBool {
        self.id.tp_gt(&other.id)
    }

    #[inline(always)]
    fn tp_gt_eq(&self, other: &AbElem) -> TpBool {
        self.id.tp_gt_eq(&other.id)
    }
}

impl TpCondSwap for AbElem {
    fn tp_cond_swap(cond: TpBool, a: &mut Self, b: &mut Self) {
        cond.cond_swap(&mut a.id, &mut b.id);
        for (a, b) in a.e11.iter_mut().zip(b.e11.iter_mut()) {
            cond.cond_swap(a, b);
        }
        cond.cond_swap(&mut a.e10, &mut b.e10);
        cond.cond_swap(&mut a.e01, &mut b.e01);
        cond.cond_swap(&mut a.e00, &mut b.e00)
    }
}
