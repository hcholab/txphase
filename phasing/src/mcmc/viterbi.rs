use crate::genotype_graph::G;
use crate::U8;
use ndarray::{Array1, Array2, ArrayView1, ArrayView3};

#[cfg(feature = "obliv")]
type Real = crate::RealHmm;

#[cfg(not(feature = "obliv"))]
type Real = f64;

#[cfg(feature = "obliv")]
pub use tp_fixedpoint::timing_shield::{TpEq, TpOrd};

pub fn viterbi(tprob_dips: ArrayView3<Real>, genotype_graph: ArrayView1<G>) -> Array2<U8> {
    let m = tprob_dips.shape()[0];
    let p = tprob_dips.shape()[1];

    #[cfg(feature = "obliv")]
    let mut backtrace = (0..m)
        .map(|_| obliv_utils::vec::OblivVec::with_capacity(p))
        .collect::<Vec<_>>();

    #[cfg(not(feature = "obliv"))]
    let mut backtrace = Array2::<U8>::zeros((m, p));

    let mut maxprob = Array1::<Real>::zeros(p);
    let mut maxprob_next = Array1::<Real>::zeros(p);

    for h1 in 0..p {
        maxprob[h1] = tprob_dips[[0, 0, h1]]; // p(x1), diploid prob
    }

    #[cfg(feature = "obliv")]
    let mut max_val = Real::protect_i64(0);

    #[cfg(not(feature = "obliv"))]
    let mut max_val = 0.;

    #[cfg(feature = "obliv")]
    let mut max_ind: U8 = U8::protect(0);

    #[cfg(not(feature = "obliv"))]
    let mut max_ind: u8 = 0;

    for i in 1..m {
        for h2 in 0..p {
            for h1 in 0..p {
                #[cfg(feature = "obliv")]
                let val = genotype_graph[i].is_segment_marker().select(
                    maxprob[h1] * tprob_dips[[i, h1, h2]],
                    if h1 == h2 {
                        maxprob[h1]
                    } else {
                        Real::protect_i64(0)
                    },
                );

                #[cfg(not(feature = "obliv"))]
                let val = if genotype_graph[i].is_segment_marker() {
                    maxprob[h1] * tprob_dips[[i, h1, h2]]
                } else {
                    if h1 == h2 {
                        maxprob[h1]
                    } else {
                        0.
                    }
                };

                if h1 == 0 {
                    max_val = val;
                    #[cfg(feature = "obliv")]
                    {
                        max_ind = U8::protect(h1 as u8);
                    }
                    #[cfg(not(feature = "obliv"))]
                    {
                        max_ind = h1 as u8;
                    }
                } else {
                    #[cfg(feature = "obliv")]
                    {
                        let cond = val.tp_gt(&max_val);
                        max_val = cond.select(val, max_val);
                        max_ind = cond.select(U8::protect(h1 as u8), max_ind);
                    }

                    #[cfg(not(feature = "obliv"))]
                    if val > max_val {
                        max_val = val;
                        max_ind = h1 as u8;
                    }
                }
            }

            maxprob_next[h2] = max_val;
            #[cfg(feature = "obliv")]
            {
                backtrace[i].push(max_ind);
            }
            #[cfg(not(feature = "obliv"))]
            {
                backtrace[[i, h2]] = max_ind;
            }
        }
        #[cfg(feature = "obliv")]
        {
            maxprob = &maxprob_next * (Real::protect_i64(1) / maxprob_next.sum());
        }

        #[cfg(not(feature = "obliv"))]
        {
            maxprob = &maxprob_next / maxprob_next.sum();
        }
    }

    // Find max value in the final vector
    for h1 in 0..p {
        let val = maxprob[h1];
        if h1 == 0 {
            max_val = val;
            #[cfg(feature = "obliv")]
            {
                max_ind = U8::protect(h1 as u8);
            }
            #[cfg(not(feature = "obliv"))]
            {
                max_ind = h1 as u8;
            }
        } else {
            #[cfg(feature = "obliv")]
            {
                let cond = val.tp_gt(&max_val);
                max_val = cond.select(val, max_val);
                max_ind = cond.select(U8::protect(h1 as u8), max_ind);
            }

            #[cfg(not(feature = "obliv"))]
            if val > max_val {
                max_val = val;
                max_ind = h1 as u8;
            }
        }
    }

    // Follow backtrace pointers to recover optimal sequence
    #[cfg(feature = "obliv")]
    let mut map_ind = Array2::<U8>::from_elem((m, 2), U8::protect(0));
    #[cfg(not(feature = "obliv"))]
    let mut map_ind = Array2::<U8>::zeros((m, 2));
    map_ind[[m - 1, 0]] = max_ind;
    map_ind[[m - 1, 1]] = (p as u8) - 1 - max_ind;
    for i in (1..m).rev() {
        #[cfg(feature = "obliv")]
        {
            max_ind = backtrace[i].get(max_ind.as_u32()); // TODO: Use linear ORAM!
        }
        #[cfg(not(feature = "obliv"))]
        {
            max_ind = backtrace[[i, max_ind as usize]]; // TODO: Use linear ORAM!
        }
        map_ind[[i - 1, 0]] = max_ind;
        map_ind[[i - 1, 1]] = (p as u8) - 1 - max_ind;
    }
    map_ind
}
