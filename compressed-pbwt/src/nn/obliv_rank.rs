use crate::{Bool, Usize};
use timing_shield::{TpBool, TpCondSwap, TpEq, TpOrd, TpU32, TpU64};

#[derive(Clone, Copy)]
pub struct NNRank((TpU64, TpU32));

impl NNRank {
    pub fn new(div: Usize, dist: Usize, is_below: Bool, hap_id: usize) -> Self {
        Self((
            div << 32 | dist << 1 | is_below.as_u64(),
            TpU32::protect(hap_id as u32),
        ))
    }

    pub fn get_hap_id(&self) -> Usize {
        self.0 .1.as_u64()
    }
}

impl Default for NNRank {
    fn default() -> Self {
        Self((TpU64::protect(0), TpU32::protect(0)))
    }
}

impl TpCondSwap for NNRank {
    fn tp_cond_swap(cond: TpBool, a: &mut Self, b: &mut Self) {
        cond.cond_swap(&mut a.0 .0, &mut b.0 .0);
        cond.cond_swap(&mut a.0 .1, &mut b.0 .1);
    }
}

impl TpEq for NNRank {
    #[inline]
    fn tp_eq(&self, rank: &NNRank) -> TpBool {
        self.0 .0.tp_eq(&rank.0 .0) & self.0 .1.tp_eq(&rank.0 .1)
    }

    #[inline]
    fn tp_not_eq(&self, rank: &NNRank) -> TpBool {
        self.0 .0.tp_not_eq(&rank.0 .0) & self.0 .1.tp_not_eq(&rank.0 .1)
    }
}

impl TpOrd for NNRank {
    #[inline]
    fn tp_lt(&self, rank: &NNRank) -> TpBool {
        self.0 .0.tp_lt(&rank.0 .0)
    }
    #[inline]
    fn tp_lt_eq(&self, rank: &NNRank) -> TpBool {
        self.0 .0.tp_lt_eq(&rank.0 .0)
    }
    #[inline]
    fn tp_gt(&self, rank: &NNRank) -> TpBool {
        self.0 .0.tp_gt(&rank.0 .0)
    }
    #[inline]
    fn tp_gt_eq(&self, rank: &NNRank) -> TpBool {
        self.0 .0.tp_gt_eq(&rank.0 .0)
    }
}
