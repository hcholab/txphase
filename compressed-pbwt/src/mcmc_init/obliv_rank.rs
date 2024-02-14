use crate::{Bool, Usize};
use timing_shield::{TpBool, TpCondSwap, TpEq, TpOrd, TpU64};

#[derive(Clone)]
pub struct InitRank(TpU64);

impl InitRank {
    pub fn new(div: Usize, dist: Usize, is_below: Bool, hap: bool) -> Self {
        Self(div << 32 | dist << 2 | is_below.as_u64() << 1 | Bool::protect(hap).as_u64())
    }

    pub fn get_div(&self) -> Usize {
        self.0 >> 32
    }

    pub fn get_hap(&self) -> Bool {
        (self.0 & 1).tp_eq(&1)
    }

    pub fn set_hap(&mut self, hap: bool) {
        if hap {
            self.0 |= 1
        } else {
            self.0 &= u64::MAX - 1;
        }
    }
}

impl Default for InitRank {
    fn default() -> Self {
        Self(TpU64::protect(0))
    }
}

impl TpCondSwap for InitRank {
    #[inline]
    fn tp_cond_swap(cond: TpBool, a: &mut Self, b: &mut Self) {
        cond.cond_swap(&mut a.0, &mut b.0);
    }
}

impl TpEq for InitRank {
    #[inline]
    fn tp_eq(&self, rank: &InitRank) -> TpBool {
        self.0.tp_eq(&rank.0)
    }
    #[inline]
    fn tp_not_eq(&self, rank: &InitRank) -> TpBool {
        self.0.tp_not_eq(&rank.0)
    }
}

impl TpOrd for InitRank {
    #[inline]
    fn tp_lt(&self, rank: &InitRank) -> TpBool {
        self.0.tp_lt(&rank.0)
    }
    #[inline]
    fn tp_lt_eq(&self, rank: &InitRank) -> TpBool {
        self.0.tp_lt_eq(&rank.0)
    }
    #[inline]
    fn tp_gt(&self, rank: &InitRank) -> TpBool {
        self.0.tp_gt(&rank.0)
    }
    #[inline]
    fn tp_gt_eq(&self, rank: &InitRank) -> TpBool {
        self.0.tp_gt_eq(&rank.0)
    }
}
