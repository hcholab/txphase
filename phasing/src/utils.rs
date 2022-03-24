#[cfg(feature = "leak-resist")]
mod inner {
    #[macro_export]
    macro_rules! tp_value {
        ($x: expr, $t: ty) => {
            paste::paste! {
                tp_fixedpoint::timing_shield::[<Tp $t:camel>]::protect($x as $t)
            }
        };
    }

    #[macro_export]
    macro_rules! tp_convert_to {
        ($x: expr, $t: ty) => {
            paste::paste! {
                $x.[<as_ $t>]()
            }
        };
    }

    #[macro_export]
    macro_rules! tp_expose {
        ($x: expr) => {
            x.expose()
        };
    }
}

#[cfg(not(feature = "leak-resist"))]
mod inner {
    #[macro_export]
    macro_rules! tp_value {
        ($x: expr, $t: ty) => {
            $x as $t
        };
    }

    #[macro_export]
    macro_rules! tp_convert_to {
        ($x: expr, $t: ty) => {
            $x as $t
        };
    }

    #[macro_export]
    macro_rules! tp_expose {
        ($x: expr) => {
            x
        };
    }
}

#[cfg(feature = "leak-resist")]
use timing_shield::{TpOrd, TpU32};

#[inline]
pub fn next_log2(v: u32) -> u32 {
    31 - v.next_power_of_two().leading_zeros()
}

#[inline]
pub fn log2(v: u32) -> u32 {
    31 - v.leading_zeros()
}

// long division algorithm
#[cfg(feature = "leak-resist")]
pub fn tp_u32_div(n: TpU32, d: TpU32, max_n: u32) -> (TpU32, TpU32) {
    let n_bits = next_log2(max_n);
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

#[cfg(feature = "leak-resist")]
pub fn tp_u32_shl(a: TpU32, b: TpU32, max_b: u32) -> TpU32 {
    let mut out = a;
    for i in 0..max_b {
        out = b.tp_gt(&i).select(out << 1, out);
    }
    out
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn div_test() {
        let max = 10000u32;
        for n in 0..max {
            let (q, r) = tp_u32_div(TpU32::protect(n), TpU32::protect(7), max);
            assert_eq!(q.expose(), n / 7);
            assert_eq!(r.expose(), n % 7);
        }
    }

    #[test]
    fn log2_test() {
        for i in 1..1000u32 {
            assert_eq!(log2(i), (i as f64).log2().floor() as u32);
            assert_eq!(next_log2(i), (i as f64).log2().ceil() as u32);
        }
    }

    #[test]
    fn tp_u32_shl_test() {
        let result = tp_u32_shl(TpU32::protect(1), TpU32::protect(3), 5);
        assert_eq!(result.expose(), 8);
    }
}
