use crate::genotype_graph::G;
use crate::{Real, U8};
use ndarray::{Array1, Array2, ArrayView1, ArrayView3};

#[cfg(feature = "leak-resist")]
mod inner {
    pub use tp_fixedpoint::timing_shield::{TpEq, TpOrd};
}

#[cfg(feature = "leak-resist")]
use inner::*;

pub fn viterbi(tprob_dips: ArrayView3<Real>, genotype_graph: ArrayView1<G>) -> Array2<U8> {
    let m = tprob_dips.shape()[0];
    let p = tprob_dips.shape()[1];

    let mut backtrace = Array2::<U8>::zeros((m, p));
    let mut maxprob = Array1::<Real>::zeros(p);
    let mut maxprob_next = Array1::<Real>::zeros(p);
    for h1 in 0..p {
        maxprob[h1] = tprob_dips[[0, 0, h1]]; // p(x1), diploid prob
    }

    let mut max_val: Real = 0.;
    let mut max_ind: U8 = 0;

    for i in 1..m {
        for h2 in 0..p {
            for h1 in 0..p {
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
                    #[cfg(feature = "leak-resist")]
                    {
                        max_ind = U8::protect(h1 as u8);
                    }
                    #[cfg(not(feature = "leak-resist"))]
                    {
                        max_ind = h1 as u8;
                    }
                } else {
                    #[cfg(feature = "leak-resist")]
                    {
                        let update = val.tp_gt(&max_val);
                        max_val = update.select(val, max_val);
                        max_ind = update.select(U8::protect(h1 as u8), max_ind);
                    }
                    #[cfg(not(feature = "leak-resist"))]
                    {
                        if val > max_val {
                            max_val = val;
                            max_ind = h1 as u8;
                        }
                    }
                }
            }

            maxprob_next[h2] = max_val;
            backtrace[[i, h2]] = max_ind;
        }

        maxprob = &maxprob_next / maxprob_next.sum();
    }

    // Find max value in the final vector
    for h1 in 0..p {
        let val = maxprob[h1];
        if h1 == 0 {
            max_val = val;
            #[cfg(feature = "leak-resist")]
            {
                max_ind = U8::protect(h1 as u8);
            }
            #[cfg(not(feature = "leak-resist"))]
            {
                max_ind = h1 as u8;
            }
        } else {
            #[cfg(feature = "leak-resist")]
            {
                let update = val.tp_gt(&max_val);
                max_val = update.select(val, max_val);
                max_ind = update.select(U8::protect(h1 as u8), max_ind);
            }
            #[cfg(not(feature = "leak-resist"))]
            {
                if val > max_val {
                    max_val = val;
                    max_ind = h1 as u8;
                }
            }
        }
    }

    // Follow backtrace pointers to recover optimal sequence
    let mut map_ind = Array2::<U8>::zeros((m, 2));
    map_ind[[m - 1, 0]] = max_ind;
    map_ind[[m - 1, 1]] = (p as u8) - 1 - max_ind;
    for i in (1..m).rev() {
        #[cfg(feature = "leak-resist")]
        {
            max_ind = backtrace[[i, max_ind.expose() as usize]]; // TODO: Use linear ORAM!
        }
        #[cfg(not(feature = "leak-resist"))]
        {
            max_ind = backtrace[[i, max_ind as usize]]; // TODO: Use linear ORAM!
        }
        map_ind[[i - 1, 0]] = max_ind;
        map_ind[[i - 1, 1]] = (p as u8) - 1 - max_ind;
    }
    map_ind
}
