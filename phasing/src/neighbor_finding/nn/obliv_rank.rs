use timing_shield::{TpBool, TpCondSwap, TpEq, TpOrd, TpU32, TpU64};

#[derive(Clone, Copy)]
#[repr(align(8))]
pub struct NNRank((TpU32, TpU32));

impl NNRank {
    pub fn new(div: TpU32, hap_id: u32) -> Self {
        Self((div, TpU32::protect(hap_id)))
    }

    pub fn get_hap_id(&self) -> TpU32 {
        self.0 .1
    }
}

impl Default for NNRank {
    fn default() -> Self {
        Self((TpU32::protect(0), TpU32::protect(0)))
    }
}

impl TpCondSwap for NNRank {
    fn tp_cond_swap(cond: TpBool, a: &mut Self, b: &mut Self) {
        let a = unsafe { (a as *mut _ as *mut TpU64).as_mut().unwrap() };
        let b = unsafe { (b as *mut _ as *mut TpU64).as_mut().unwrap() };
        cond.cond_swap(a, b);
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
