use crate::{Real, U8};
use ndarray::{Array1, Array2, ArrayView3};

#[cfg(feature = "leak-resist")]
mod inner {
    pub use tp_fixedpoint::timing_shield::{TpEq, TpOrd};
}

#[cfg(feature = "leak-resist")]
use inner::*;

// Takes in p(x1), p(x2|x1), ..., p(xn|xn-1)
// Outputs most likely (paired) assignment using Viterbi algorithm
pub fn viterbi(tprob: ArrayView3<Real>) -> Array2<U8> {
    println!("{:?}", tprob);
    let m = tprob.shape()[0];
    let p = tprob.shape()[1];

    let mut backtrace = unsafe { Array2::<U8>::uninit((m, p)).assume_init() };
    let mut maxprob = unsafe { Array1::<Real>::uninit(p).assume_init() };
    let mut maxprob_next = unsafe { Array1::<Real>::uninit(p).assume_init() };
    for h1 in 0..p {
        maxprob[h1] = tprob[[0, 0, h1]] * tprob[[0, 0, p - 1 - h1]]; // p(x1), diploid prob
    }

    let mut max_val: Real = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
    let mut max_ind: U8 = unsafe { std::mem::MaybeUninit::uninit().assume_init() };

    for i in 1..m {
        for h2 in 0..p {
            for h1 in 0..p {
                let val = maxprob[h1] * tprob[[i, h1, h2]] * tprob[[i, p - 1 - h1, p - 1 - h2]]; // diploid
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

        let sum: Real = maxprob_next.iter().sum();

        maxprob = &maxprob_next / sum;
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
    let mut map_ind = unsafe { Array2::<U8>::uninit((2, m)).assume_init() };
    map_ind[[0, m - 1]] = max_ind;
    map_ind[[1, m - 1]] = (p as u8) - 1 - max_ind;
    for i in (1..m).rev() {
        #[cfg(feature = "leak-resist")]
        {
            max_ind = backtrace[[i, max_ind.expose() as usize]]; // TODO: Use linear ORAM!
        }
        #[cfg(not(feature = "leak-resist"))]
        {
            max_ind = backtrace[[i, max_ind as usize]]; // TODO: Use linear ORAM!
        }
        map_ind[[0, i - 1]] = max_ind;
        map_ind[[1, i - 1]] = (p as u8) - 1 - max_ind;
    }

    map_ind
}

#[cfg(test)]
mod ref_alg {
    pub fn viterbi(tprob: &Vec<Vec<Vec<f32>>>) -> Vec<Vec<u8>> {
        let m = tprob.len();
        let p = tprob[0].len();

        let mut backtrace = vec![vec![0 as u8; p]; m];
        let mut maxprob_next = vec![0.0 as f32; p];
        let mut maxprob = vec![0.0 as f32; p];
        for h1 in 0..p {
            maxprob[h1] = tprob[0][0][h1] * tprob[0][0][p - 1 - h1]; // p(x1), diploid prob
        }

        for i in 1..m {
            for h2 in 0..p {
                let mut max_val = f32::NEG_INFINITY;
                let mut max_ind = 0 as u8;

                for h1 in 0..p {
                    let val = maxprob[h1] * tprob[i][h1][h2] * tprob[i][p - 1 - h1][p - 1 - h2]; // diploid
                    let update = val > max_val;
                    max_val = mutex2f32(update, val, max_val);
                    max_ind = mutex2u8(update, h1 as u8, max_ind);
                }

                maxprob_next[h2] = max_val;
                backtrace[i][h2] = max_ind;
            }

            // TODO: replace with lazy renormalization based on shifts
            let sum = sum_vec(maxprob_next.as_slice());
            for h2 in 0..p {
                maxprob[h2] = maxprob_next[h2] / sum;
            }
        }

        // Find max value in the final vector
        let mut max_val = f32::NEG_INFINITY;
        let mut max_ind = 0 as u8;
        for h1 in 0..p {
            let val = maxprob[h1];
            let update = val > max_val;
            max_val = mutex2f32(update, val, max_val);
            max_ind = mutex2u8(update, h1 as u8, max_ind);
        }

        // Follow backtrace pointers to recover optimal sequence
        let mut map_ind = vec![vec![0 as u8; m]; 2];
        map_ind[0][m - 1] = max_ind;
        map_ind[1][m - 1] = (p as u8) - 1 - max_ind;
        for i in (1..m).rev() {
            max_ind = backtrace[i][max_ind as usize]; // TODO: Use linear ORAM!
            map_ind[0][i - 1] = max_ind;
            map_ind[1][i - 1] = (p as u8) - 1 - max_ind;
        }

        map_ind
    }
    fn mutex2u8(b: bool, v1: u8, v0: u8) -> u8 {
        if b {
            return v1;
        } else {
            return v0;
        }
    }

    fn mutex2f32(b: bool, v1: f32, v0: f32) -> f32 {
        if b {
            return v1;
        } else {
            return v0;
        }
    }

    fn sum_vec(v: &[f32]) -> f32 {
        let mut out = 0.0;
        for i in 0..v.len() {
            out += v[i]
        }
        out
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::Array3;
    use rand::Rng;
    use std::collections::HashSet;

    #[test]
    fn viterbi_test() {
        let m = 1000;
        let p = 4;
        let mut rng = rand::thread_rng();

        let ref_tprob = (0..m)
            .map(|_| {
                (0..p)
                    .map(|_| (0..p).map(|_| rng.gen_range(0.0..1.0)).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        #[cfg(feature = "leak-resist")]
        let tprob = Array3::from_shape_fn((m, p, p), |(i, j, k)| {
            Real::protect_f32(ref_tprob[i][j][k])
        });

        #[cfg(not(feature = "leak-resist"))]
        let tprob = Array3::from_shape_fn((m, p, p), |(i, j, k)| ref_tprob[i][j][k]);

        let ref_results = ref_alg::viterbi(&ref_tprob)
            .into_iter()
            .collect::<HashSet<_>>();

        let results = viterbi(tprob.view());

        #[cfg(feature = "leak-resist")]
        let results = Array2::from_shape_fn((2, m), |(i, j)| results[[i, j]].expose());

        let results = (0..2)
            .map(|i| (0..m).map(|j| results[[i, j]]).collect::<Vec<_>>())
            .collect::<HashSet<_>>();

        assert_eq!(results, ref_results);
    }
}
