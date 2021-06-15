use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis, Zip};
use std::time::Instant;
#[cfg(feature = "leak-resist")]
mod inner {
    pub use tp_fixedpoint::timing_shield::{TpBool, TpEq, TpU8};
    pub type Real = tp_fixedpoint::TpLnFixed<20>;
    pub type Genotype = TpU8;
    pub use tp_fixedpoint::num_traits::Zero;
}

#[cfg(not(feature = "leak-resist"))]
mod inner {
    pub type Real = f32;
    pub type Genotype = u8;
}

// random initialization
fn gen_data(m: usize, n: usize) -> (Vec<Vec<u8>>, Vec<u8>) {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
    let mut x = vec![vec![0 as u8; n]; m]; // ref panel
    let mut t = vec![0 as u8; m]; // target
    for i in 0..m {
        for j in 0..n {
            if i % 10 > 0 {
                x[i][j] = x[i - 1][j]
            } else {
                x[i][j] = rng.gen_range(0..2);
            }
        }
        t[i] = rng.gen_range(0..2); // 0, 1, or 2
    }
    (x, t)
}

pub fn test_hmm() {
    let m = 1000; // num of variants
    let n = 100; // num of samples
    let (x, t) = gen_data(m, n);
    let ref_result = Array1::from_vec(hmm_ref(&x, &t));
    #[cfg(feature = "leak-resist")]
    let (x, t) = {
        let x = x
            .into_iter()
            .map(|vec| vec.into_iter().map(|v| TpU8::protect(v)).collect())
            .collect::<Vec<_>>();
        let t = t.into_iter().map(|v| TpU8::protect(v)).collect::<Vec<_>>();
        (x, t)
    };
    let result = hmm(&x, &t);
    #[cfg(feature = "leak-resist")]
    let result = Array1::from_vec(
        result
            .into_iter()
            .map(|v| {
                if v.leaky_is_nan() {
                    -1.0
                } else {
                    v.leaky_into_f32()
                }
            })
            .collect::<Vec<_>>(),
    );
    //let mut sum_error = 0.0;
    //for (&i, &r) in result.iter().zip(ref_result.iter()) {
    //let i: f32 = i.into();
    //sum_error += (i - r).abs();
    ////println!("{}", (i-r).abs());
    //assert!((i - r).abs() < 0.01);
    //}
    //println!("total sum error = {}", sum_error);
    assert!(result.abs_diff_eq(&ref_result, 0.1));
    //assert!(result.abs_diff_eq(&ref_result, 0.0001));
}

use inner::*;

fn get_emission(
    x_col: ArrayView1<Genotype>,
    t: Genotype,
    eprob: Real,
    neg_eprob: Real,
) -> Array2<Real> {
    let n = x_col.len();
    let mut eprobs = unsafe { Array2::<Real>::uninit((n, n)).assume_init() };
    Zip::from(eprobs.outer_iter_mut())
        .and(x_col)
        .for_each(|mut ps, &g1| {
            // prob of observing 1 in each haplotype
            #[cfg(feature = "leak-resist")]
            let (e1, neg_e1) = {
                let mut eprob = eprob;
                let mut neg_eprob = neg_eprob;
                g1.tp_eq(&1).cond_swap(&mut eprob, &mut neg_eprob);
                (eprob, neg_eprob)
            };

            #[cfg(not(feature = "leak-resist"))]
            let e1 = if g1 == 1 { neg_eprob } else { eprob };

            Zip::from(&mut ps).and(x_col).for_each(|p, &g2| {
                // prob of observing 1 in each haplotype
                #[cfg(feature = "leak-resist")]
                {
                    let (mut e2, mut neg_e2) = (eprob, neg_eprob);
                    g2.tp_eq(&1).cond_swap(&mut e2, &mut neg_e2);
                    let dos0 = neg_e1 * neg_e2;
                    let dos1 = e1 * neg_e2 + e2 * neg_e1;
                    let dos2 = e1 * e2;
                    *p = t.tp_eq(&0).select(dos0, t.tp_eq(&1).select(dos1, dos2));
                }

                #[cfg(not(feature = "leak-resist"))]
                {
                    let e2 = if g2 == 1 { neg_eprob } else { eprob };

                    // distribution over dosage (0, 1, 2)
                    let mut dos: [f32; 3] = [0.0; 3];
                    dos[0] = (1.0 - e1) * (1.0 - e2);
                    dos[1] = e1 * (1.0 - e2) + e2 * (1.0 - e1);
                    dos[2] = e1 * e2;

                    // select based on target and multiply
                    *p = dos[t as usize];
                }
            });
        });
    eprobs
}

fn get_transition(
    cur: ArrayView2<Real>,
    rprob: Real,
    neg_rprob: Real,
    n_real: Real,
    n2_real: Real,
) -> Array2<Real> {
    // compute row/col/total sums
    #[cfg(not(feature = "leak-resist"))]
    let (mut psum, mut msum, totsum) = {
        let psum = cur.sum_axis(Axis(0));
        let msum = cur.sum_axis(Axis(1));
        let totsum = psum.sum();
        (psum, msum, totsum)
    };

    #[cfg(feature = "leak-resist")]
    let (mut psum, mut msum, totsum) = {
        let n = cur.nrows();
        let mut psum = unsafe { Array1::<Real>::uninit(n).assume_init() };
        Zip::from(cur.columns())
            .and(&mut psum)
            .for_each(|row, sum| {
                *sum = row.into_iter().sum();
            });

        let mut msum = unsafe { Array1::<Real>::uninit(n).assume_init() };
        Zip::from(cur.rows()).and(&mut msum).for_each(|col, sum| {
            *sum = col.into_iter().sum();
        });
        let totsum = msum.iter().sum::<Real>();
        (psum, msum, totsum)
    };

    // multiply transition prob and set new fprob
    let mut rprobs = &cur * neg_rprob * neg_rprob + totsum * rprob * rprob / n2_real;
    psum *= neg_rprob * rprob / n_real;
    msum *= neg_rprob * rprob / n_real;

    Zip::from(rprobs.outer_iter_mut())
        .and(&msum)
        .for_each(|fps, &m| {
            Zip::from(fps).and(&psum).for_each(|fp, &p| *fp += m + p);
        });

    rprobs
}

fn phase_target(
    cur_x: ArrayView1<Genotype>,
    t: Genotype,
    cur_forward: ArrayView2<Real>,
    cur_backward: ArrayView2<Real>,
) -> Real {
    let phase: Real;
    #[cfg(feature = "leak-resist")]
    {
        // combine fprob and prob
        let fb = &cur_forward * &cur_backward;

        let mut phase0_filtered = Vec::with_capacity(fb.len());
        let mut phase1_filtered = Vec::with_capacity(fb.len());

        Zip::from(fb.outer_iter()).and(&cur_x).for_each(|ps, &g1| {
            Zip::from(ps).and(&cur_x).for_each(|&p, &g2| {
                let cond0 = g1.tp_eq(&1) & g2.tp_eq(&0);
                let cond1 = g1.tp_eq(&0) & g2.tp_eq(&1);
                phase0_filtered.push(cond0.select(p, Real::NAN));
                phase1_filtered.push(cond1.select(p, Real::NAN));
            });
        });

        let phase0_prob = Real::checked_sum_in_place(&mut phase0_filtered);
        let phase1_prob = Real::checked_sum_in_place(&mut phase1_filtered);
        phase = t
            .tp_eq(&1)
            .select(phase1_prob / (phase0_prob + phase1_prob), Real::NAN); // fix
    }

    #[cfg(not(feature = "leak-resist"))]
    {
        // het site
        if t == 1 {
            let mut phase0_prob = 0.0;
            let mut phase1_prob = 0.0;
            // combine fprob and prob
            let fb = &cur_forward * &cur_backward;

            Zip::from(fb.outer_iter()).and(&cur_x).for_each(|ps, &g1| {
                Zip::from(ps).and(&cur_x).for_each(|&p, &g2| {
                    if g1 == 1 && g2 == 0 {
                        phase0_prob += p;
                    } else if g1 == 0 && g2 == 1 {
                        phase1_prob += p;
                    }
                });
            });
            phase = phase1_prob / (phase0_prob + phase1_prob);
        } else {
            phase = -1.0;
        }
    }

    phase
}

pub fn hmm(_x: &[Vec<Genotype>], _t: &[Genotype]) -> Array1<Real> {
    let m = _x.len(); // num of variants
    let n = _x[0].len(); // num of samples
    let n2 = n * n; // num of states (haplotype pair)

    let n_real: Real = (n as f32).into();
    let n2_real: Real = (n2 as f32).into();

    let mut x = unsafe { Array2::<Genotype>::uninit((m, n)).assume_init() };
    for (x_i, _x_i) in x.iter_mut().zip(_x.iter().flatten()) {
        *x_i = *_x_i;
    }

    let t = Array1::from_vec(_t.to_vec()); // target

    let mut phase: Array1<Real> = Array1::from_elem(m, Into::<Real>::into(-1.0f32)); // output (phase prob over het sites)

    // hmm parameters
    let rprob = 0.05; // recombination
    let neg_rprob = 1.0 - rprob;
    let eprob = 0.001; // error
    let neg_eprob = 1.0 - eprob;

    let rprob: Real = rprob.into(); // recombination
    let neg_rprob: Real = neg_rprob.into();
    let eprob: Real = eprob.into(); // error
    let neg_eprob: Real = neg_eprob.into();

    let start = Instant::now();

    let mut fprob = unsafe { Array3::<Real>::uninit((m, n, n)).assume_init() }; // forward prob

    let two: Real = 2.0.into();
    let one: Real = 1.0.into();

    #[cfg(feature = "leak-resist")]
    let zero: Real = Real::NAN;

    #[cfg(not(feature = "leak-resist"))]
    let zero = 0.0;

    // initialization (uniform + symmetry breaking)
    let mut first_fprob = fprob.index_axis_mut(Axis(0), 0);
    for h1 in 0..n {
        for h2 in 0..n {
            if h1 > h2 {
                first_fprob[[h1, h2]] = two;
            } else if h1 == h2 {
                first_fprob[[h1, h2]] = one;
            } else {
                first_fprob[[h1, h2]] = zero;
            }
        }
    }

    // forward pass
    let mut cur = unsafe { Array2::<Real>::uninit((n, n)).assume_init() };
    for i in 1..m {
        // copy previous fprob
        cur.assign(&fprob.index_axis(Axis(0), i - 1));

        // multiply emission prob
        cur = cur * get_emission(x.row(i), t[i - 1], eprob, neg_eprob);

        // multiply transition prob and set new fprob
        let mut cur_fprob = fprob.index_axis_mut(Axis(0), i);
        cur_fprob.assign(&get_transition(
            cur.view(),
            rprob,
            neg_rprob,
            n_real,
            n2_real,
        ));

        // renormalize
        #[cfg(feature = "leak-resist")]
        let sum = cur_fprob.iter().sum::<Real>();

        #[cfg(not(feature = "leak-resist"))]
        let sum = cur_fprob.sum();

        cur_fprob /= sum;
    }

    // backward pass + output
    // last position
    cur = get_emission(x.row(m - 1), t[m - 1], eprob, neg_eprob);

    phase[m - 1] = phase_target(
        x.row(m - 1),
        t[m - 1],
        fprob.index_axis(Axis(0), m - 1),
        cur.view(),
    );

    for i in (0..m - 1).rev() {
        cur.assign(&get_transition(
            cur.view(),
            rprob,
            neg_rprob,
            n_real,
            n2_real,
        ));

        // multiply emission prob
        cur = cur * get_emission(x.row(i), t[i], eprob, neg_eprob);

        // renormalize
        #[cfg(feature = "leak-resist")]
        let sum = cur.iter().sum::<Real>();

        #[cfg(not(feature = "leak-resist"))]
        let sum = cur.sum();

        cur /= sum;

        let cur_fprob = fprob.index_axis(Axis(0), i);
        phase[i] = phase_target(x.row(i), t[i], cur_fprob.view(), cur.view());
    }

    println!("time = {} ms", (Instant::now() - start).as_millis());

    phase
}

fn hmm_ref(x: &[Vec<u8>], t: &[u8]) -> Vec<f32> {
    let m = x.len(); // num of variants
    let n = x[0].len(); // num of samples
    let n2 = n * n; // num of states (haplotype pair)

    let mut phase = vec![-1.0 as f32; m]; // output (phase prob over het sites)

    // hmm parameters
    let rprob = 0.05; // recombination
    let eprob = 0.001; // error

    let start = Instant::now();

    let mut fprob = vec![vec![vec![0.0 as f32; n]; n]; m]; // forward prob

    // initialization (uniform + symmetry breaking)
    for h1 in 0..n {
        for h2 in 0..n {
            if h1 > h2 {
                fprob[0][h1][h2] = 2.0;
            } else if h1 == h2 {
                fprob[0][h1][h2] = 1.0;
            } else {
                fprob[0][h1][h2] = 0.0;
            }
        }
    }

    // forward pass
    let mut cur = vec![vec![0.0 as f32; n]; n];
    for i in 1..m {
        // copy previous fprob
        for h1 in 0..n {
            for h2 in 0..n {
                cur[h1][h2] = fprob[i - 1][h1][h2];
            }
        }

        // multiply emission prob
        for h1 in 0..n {
            for h2 in 0..n {
                // prob of observing 1 in each haplotype
                let g1 = if x[i][h1] == 1 { 1.0 - eprob } else { eprob };
                let g2 = if x[i][h2] == 1 { 1.0 - eprob } else { eprob };

                // distribution over dosage (0, 1, 2)
                let mut dos: [f32; 3] = [0.0; 3];
                dos[0] = (1.0 - g1) * (1.0 - g2);
                dos[1] = g1 * (1.0 - g2) + g2 * (1.0 - g1);
                dos[2] = g1 * g2;

                // select based on target and multiply
                cur[h1][h2] *= dos[t[i - 1] as usize];
            }
        }

        // compute row/col/total sums
        let mut msum = vec![0 as f32; n];
        let mut psum = vec![0 as f32; n];
        let mut totsum = 0 as f32;
        for h1 in 0..n {
            for h2 in 0..n {
                msum[h1] += cur[h1][h2];
                psum[h2] += cur[h1][h2];
                totsum += cur[h1][h2];
            }
        }

        // multiply transition prob and set new fprob
        let mut sum = 0 as f32;
        for h1 in 0..n {
            for h2 in 0..n {
                fprob[i][h1][h2] = cur[h1][h2] * (1.0 - rprob) * (1.0 - rprob)
                    + msum[h1] * (1.0 - rprob) * rprob / (n as f32)
                    + psum[h2] * (1.0 - rprob) * rprob / (n as f32)
                    + totsum * rprob * rprob / (n2 as f32);

                sum += fprob[i][h1][h2];
            }
        }

        // renormalize
        for h1 in 0..n {
            for h2 in 0..n {
                fprob[i][h1][h2] /= sum;
            }
        }
    }

    // backward pass + output
    // last position
    let mut phase0_prob = 0 as f32;
    let mut phase1_prob = 0 as f32;

    for h1 in 0..n {
        for h2 in 0..n {
            // prob of observing 1 in each haplotype
            let g1 = if x[m - 1][h1] == 1 {
                1.0 - eprob
            } else {
                eprob
            };
            let g2 = if x[m - 1][h2] == 1 {
                1.0 - eprob
            } else {
                eprob
            };

            // distribution over dosage (0, 1, 2)
            let mut dos: [f32; 3] = [0.0; 3];
            dos[0] = (1.0 - g1) * (1.0 - g2);
            dos[1] = g1 * (1.0 - g2) + g2 * (1.0 - g1);
            dos[2] = g1 * g2;

            // select based on target
            cur[h1][h2] = dos[t[m - 1] as usize];

            if t[m - 1] == 1 {
                // het site
                // combine fprob and prob
                let fb = fprob[m - 1][h1][h2] * cur[h1][h2];
                if x[m - 1][h1] == 1 && x[m - 1][h2] == 0 {
                    phase0_prob += fb;
                } else if x[m - 1][h1] == 0 && x[m - 1][h2] == 1 {
                    phase1_prob += fb;
                }
            }
        }
    }

    if t[m - 1] == 1 {
        phase[m - 1] = phase1_prob / (phase0_prob + phase1_prob);
    }

    for i in (0..m - 1).rev() {
        // compute row/col/total sums
        let mut msum = vec![0.0 as f32; n];
        let mut psum = vec![0.0 as f32; n];
        let mut totsum = 0.0 as f32;
        for h1 in 0..n {
            for h2 in 0..n {
                msum[h1] += cur[h1][h2];
                psum[h2] += cur[h1][h2];
                totsum += cur[h1][h2];
            }
        }

        // multiply transition prob
        for h1 in 0..n {
            for h2 in 0..n {
                cur[h1][h2] = cur[h1][h2] * (1.0 - rprob) * (1.0 - rprob)
                    + msum[h1] * (1.0 - rprob) * rprob / (n as f32)
                    + psum[h2] * (1.0 - rprob) * rprob / (n as f32)
                    + totsum * rprob * rprob / (n2 as f32);
            }
        }

        // multiply emission prob
        let mut sum = 0.0 as f32;
        for h1 in 0..n {
            for h2 in 0..n {
                // prob of observing 1 in each haplotype
                let g1 = if x[i][h1] == 1 { 1.0 - eprob } else { eprob };
                let g2 = if x[i][h2] == 1 { 1.0 - eprob } else { eprob };

                // distribution over dosage (0, 1, 2)
                let mut dos: [f32; 3] = [0.0; 3];
                dos[0] = (1.0 - g1) * (1.0 - g2);
                dos[1] = g1 * (1.0 - g2) + g2 * (1.0 - g1);
                dos[2] = g1 * g2;

                // select based on target and multiply
                cur[h1][h2] *= dos[t[i] as usize];

                sum += cur[h1][h2];
            }
        }

        // renormalize
        for h1 in 0..n {
            for h2 in 0..n {
                cur[h1][h2] /= sum;
            }
        }

        if t[i] == 1 {
            // het site
            phase0_prob = 0.0;
            phase1_prob = 0.0;

            // combine fprob and prob
            for h1 in 0..n {
                for h2 in 0..n {
                    let fb = fprob[i][h1][h2] * cur[h1][h2];
                    if x[i][h1] == 1 && x[i][h2] == 0 {
                        phase0_prob += fb;
                    } else if x[i][h1] == 0 && x[i][h2] == 1 {
                        phase1_prob += fb;
                    }
                }
            }

            phase[i] = phase1_prob / (phase0_prob + phase1_prob);
        }
    }

    println!("time = {} ms", (Instant::now() - start).as_millis());

    phase
}
