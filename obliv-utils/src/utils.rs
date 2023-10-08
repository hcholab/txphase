use timing_shield::{TpOrd, TpU16, TpU32, TpU64};

pub fn tp_u16_div(n: TpU16, d: TpU16, max_n: u16) -> (TpU16, TpU16) {
    let n_bits = next_log2_u16(max_n);
    let mut q = TpU16::protect(0);
    let mut r = TpU16::protect(0);
    for i in (0..n_bits).rev() {
        r <<= 1;
        r |= (n >> i) & 1;
        let cond = r.tp_gt_eq(&d);
        r = cond.select(r - d, r);
        q = cond.select(q | (1 << i), q);
    }
    (q, r)
}

#[inline]
fn next_log2_u16(v: u16) -> u32 {
    15 - v.next_power_of_two().leading_zeros()
}

pub fn tp_u32_div(n: TpU32, d: TpU32, max_n: u32) -> (TpU32, TpU32) {
    let n_bits = next_log2_u32(max_n);
    let mut q = TpU32::protect(0);
    let mut r = TpU32::protect(0);
    for i in (0..n_bits).rev() {
        r <<= 1;
        r |= (n >> i) & 1;
        let cond = r.tp_gt_eq(&d);
        r = cond.select(r - d, r);
        q = cond.select(q | (1 << i), q);
    }
    (q, r)
}

#[inline]
fn next_log2_u32(v: u32) -> u32 {
    31 - v.next_power_of_two().leading_zeros()
}

pub fn tp_u64_div(n: TpU64, d: TpU64, max_n: u64) -> (TpU64, TpU64) {
    let n_bits = next_log2_u64(max_n);
    let mut q = TpU64::protect(0);
    let mut r = TpU64::protect(0);
    for i in (0..n_bits).rev() {
        r <<= 1;
        r |= (n >> i) & 1;
        let cond = r.tp_gt_eq(&d);
        r = cond.select(r - d, r);
        q = cond.select(q | (1 << i), q);
    }
    (q, r)
}

#[inline]
fn next_log2_u64(v: u64) -> u32 {
    63 - v.next_power_of_two().leading_zeros()
}
