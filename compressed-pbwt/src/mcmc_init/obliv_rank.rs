use timing_shield::{TpBool, TpCondSwap, TpEq, TpOrd, TpU32};

#[derive(Clone)]
pub struct InitRank(TpU32);

impl InitRank {
    pub fn new(div: TpU32, hap: bool) -> Self {
        Self(div << 1 | hap as u32)
    }

    pub fn get_div(&self) -> TpU32 {
        self.0 >> 1
    }

    pub fn get_hap(&self) -> TpBool {
        (self.0 & 1).tp_eq(&1)
    }

    pub fn set_hap(&mut self, hap: bool) {
        if hap {
            self.0 |= 1
        } else {
            self.0 &= u32::MAX - 1;
        }
    }
}

impl Default for InitRank {
    fn default() -> Self {
        Self(TpU32::protect(0))
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
        (self.0 >> 1).tp_lt(&(rank.0 >> 1))
    }
    #[inline]
    fn tp_lt_eq(&self, rank: &InitRank) -> TpBool {
        (self.0 >> 1).tp_lt_eq(&(rank.0 >> 1))
    }
    #[inline]
    fn tp_gt(&self, rank: &InitRank) -> TpBool {
        (self.0 >> 1).tp_gt(&(rank.0 >> 1))
    }
    #[inline]
    fn tp_gt_eq(&self, rank: &InitRank) -> TpBool {
        (self.0 >> 1).tp_gt_eq(&(rank.0 >> 1))
    }
}
