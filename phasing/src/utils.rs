use rand::Rng;

//#[macro_export]
//macro_rules! branch {
    //{
        //if $($rest:tt)*
    //} => {
        //branch_parser! {
            //predicate = ()
            //rest = ($($rest)*)
        //}
    //};
//}

//#[macro_export]
//macro_rules! branch_parser {
    //{
        //predicate = ($($predicate:tt)*)
        //rest = ({ $($then:tt)* } else { $($else:tt)* })
    //} => {
        //println!("predicate: {}", stringify!($($predicate)*));
        //println!("then: {}", stringify!($($then)*));
        //println!("else: {}", stringify!($($else)*));
    //};

    //{
        //predicate = ($($predicate:tt)*)
        //rest = ($next:tt $($rest:tt)*)
    //} => {
        //branch_parser! {
            //predicate = ($($predicate)* $next)
            //rest = ($($rest)*)
        //}
    //};
//}

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

pub fn gen_ref_panel(n_samples: usize, n_markers: usize, mut rng: impl Rng) -> Vec<Vec<u8>> {
    (0..n_markers)
        .into_iter()
        .map(|_| {
            (0..n_samples)
                .into_iter()
                .map(|_| rng.gen_range(0..2))
                .collect()
        })
        .collect()
}

pub fn gen_target_sample(n_markers: usize, mut rng: impl Rng) -> Vec<u8> {
    (0..n_markers)
        .into_iter()
        .map(|_| rng.gen_range(0..2))
        .collect()
}
