#[cfg(feature = "obliv")]
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
    macro_rules! tp_value_real {
        ($x: expr, $t: ty) => {
            paste::paste! {
                crate::Real::[<protect_ $t>](($x) as $t)
            }
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
            $x
        };
    }

    #[macro_export]
    macro_rules! tp_expose_real {
        ($x: expr) => {
            $x.expose_into_f32()
        };
    }
}

#[cfg(not(feature = "obliv"))]
mod inner {
    #[macro_export]
    macro_rules! tp_value {
        ($x: expr, $t: ty) => {
            $x as $t
        };
    }

    #[macro_export]
    macro_rules! tp_value_real {
        ($x: expr, $t: ty) => {
            $x as f64
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
            $x
        };
    }

    #[macro_export]
    macro_rules! tp_expose_real {
        ($x: expr) => {
            $x
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
