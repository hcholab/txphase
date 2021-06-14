use ndarray::{Array1, Array2, Array3, Axis, Zip};
use std::time::Instant;

pub fn test_hmm() {
    let m = 10000; // num of variants
    let n = 100; // num of samples
    let (x, t) = gen_data(m, n);
    let result = hmm(&x, &t);
    let ref_result = Array1::from_vec(hmm_ref(&x, &t));
    assert!(result.abs_diff_eq(&ref_result, 0.00001));
}

pub fn hmm(_x: &[Vec<u8>], _t: &[u8]) -> Array1<f32> {
    let m = _x.len(); // num of variants
    let n = _x[0].len(); // num of samples
    let n2 = n * n; // num of states (haplotype pair)

    let mut x = Array2::<u8>::zeros((m, n));
    for (x_i, _x_i) in x.iter_mut().zip(_x.iter().flatten()) {
        *x_i = *_x_i;
    }

    let t = Array1::from_vec(_t.to_vec()); // target

    let mut phase = Array1::from_elem(m, -1.0f32); // output (phase prob over het sites)

    // hmm parameters
    let rprob = 0.05; // recombination
    let neg_rprob = 1.0 - rprob;
    let eprob = 0.001; // error
    let neg_eprob = 1.0 - eprob;

    let start = Instant::now();

    let mut fprob = Array3::<f32>::zeros((m, n, n)); // forward prob

    // initialization (uniform + symmetry breaking)
    let mut first_fprob = fprob.index_axis_mut(Axis(0), 0);
    for h1 in 0..n {
        for h2 in 0..n {
            if h1 > h2 {
                first_fprob[[h1, h2]] = 2.0;
            } else if h1 == h2 {
                first_fprob[[h1, h2]] = 1.0;
            } else {
                first_fprob[[h1, h2]] = 0.0;
            }
        }
    }

    // forward pass
    let mut cur = Array2::<f32>::zeros((n, n));
    for i in 1..m {
        // copy previous fprob
        cur.assign(&fprob.index_axis(Axis(0), i - 1));

        // multiply emission prob
        let cur_x = x.row(i);
        let last_t = t[i - 1] as usize;
        Zip::from(cur.outer_iter_mut())
            .and(&cur_x)
            .for_each(|mut ps, &g1| {
                // prob of observing 1 in each haplotype
                let e1 = if g1 == 1 { neg_eprob } else { eprob };

                Zip::from(&mut ps).and(&cur_x).for_each(|p, &g2| {
                    // prob of observing 1 in each haplotype
                    let e2 = if g2 == 1 { neg_eprob } else { eprob };

                    // distribution over dosage (0, 1, 2)
                    let mut dos: [f32; 3] = [0.0; 3];
                    dos[0] = (1.0 - e1) * (1.0 - e2);
                    dos[1] = e1 * (1.0 - e2) + e2 * (1.0 - e1);
                    dos[2] = e1 * e2;

                    // select based on target and multiply
                    *p *= dos[last_t];
                });
            });

        // compute row/col/total sums
        let mut psum = cur.sum_axis(Axis(0));
        let mut msum = cur.sum_axis(Axis(1));
        let mut totsum = psum.sum();

        // multiply transition prob and set new fprob
        cur *= neg_rprob * neg_rprob;
        totsum *= rprob * rprob / n2 as f32;
        cur += totsum;
        psum *= neg_rprob * rprob / n as f32;
        msum *= neg_rprob * rprob / n as f32;

        Zip::from(cur.outer_iter_mut())
            .and(&msum)
            .for_each(|fps, m| {
                Zip::from(fps).and(&psum).for_each(|fp, p| *fp += m + p);
            });

        let mut cur_fprob = fprob.index_axis_mut(Axis(0), i);
        cur_fprob.assign(&cur);

        // renormalize
        let sum = cur_fprob.sum();
        cur_fprob /= sum;
    }

    // backward pass + output
    // last position
    let mut phase0_prob = 0f32;
    let mut phase1_prob = 0f32;

    let last_x = x.index_axis(Axis(0), m-1);
    for h1 in 0..n {
        for h2 in 0..n {
            // prob of observing 1 in each haplotype
            let g1 = if last_x[[h1]] == 1 {
                neg_eprob
            } else {
                eprob
            };
            let g2 = if last_x[[h2]] == 1 {
                neg_eprob
            } else {
                eprob
            };

            // distribution over dosage (0, 1, 2)
            let mut dos: [f32; 3] = [0.0; 3];
            dos[0] = (1.0 - g1) * (1.0 - g2);
            dos[1] = g1 * (1.0 - g2) + g2 * (1.0 - g1);
            dos[2] = g1 * g2;

            // select based on target
            cur[[h1, h2]] = dos[t[m - 1] as usize];

            if t[m - 1] == 1 {
                // het site
                // combine fprob and prob
                let fb = fprob[[m - 1, h1, h2]] * cur[[h1, h2]];
                if last_x[[h1]] == 1 && last_x[[h2]] == 0 {
                    phase0_prob += fb;
                } else if last_x[[h1]] == 0 && last_x[[h2]] == 1 {
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
        let mut psum = cur.sum_axis(Axis(0));
        let mut msum = cur.sum_axis(Axis(1));
        let mut totsum = psum.sum();

        // multiply transition prob
        cur *= neg_rprob * neg_rprob;
        totsum *= rprob * rprob / n2 as f32;
        cur += totsum;
        psum *= neg_rprob * rprob / n as f32;
        msum *= neg_rprob * rprob / n as f32;

        Zip::from(cur.outer_iter_mut())
            .and(&msum)
            .for_each(|fps, m| {
                Zip::from(fps).and(&psum).for_each(|fp, p| *fp += m + p);
            });

        // multiply emission prob
        let cur_x = x.row(i);
        let mut sum = 0.0 as f32;
        let cur_t = t[i] as usize;
        Zip::from(cur.outer_iter_mut())
            .and(&cur_x)
            .for_each(|mut ps, &g1| {
                // prob of observing 1 in each haplotype
                let e1 = if g1 == 1 { neg_eprob } else { eprob };
                Zip::from(&mut ps).and(&cur_x).for_each(|p, &g2| {
                    // prob of observing 1 in each haplotype
                    let e2 = if g2 == 1 { neg_eprob } else { eprob };

                    // distribution over dosage (0, 1, 2)
                    let mut dos: [f32; 3] = [0.0; 3];
                    dos[0] = (1.0 - e1) * (1.0 - e2);
                    dos[1] = e1 * (1.0 - e2) + e2 * (1.0 - e1);
                    dos[2] = e1 * e2;
                    // select based on target and multiply
                    *p *= dos[cur_t];

                    sum += *p;
                });
            });

        // renormalize
        cur /= sum;

        let cur_fprob = fprob.index_axis(Axis(0), i);

        if t[i] == 1 {

            // het site
            phase0_prob = 0.0;
            phase1_prob = 0.0;

            // combine fprob and prob
            let fb = &cur_fprob * &cur;
            let cur_x = x.index_axis(Axis(0), i);

            Zip::from(fb.outer_iter()).and(&cur_x).for_each(|ps, &g1| {
                Zip::from(ps).and(&cur_x).for_each(|&p, &g2| {
                    if g1 == 1 && g2 == 0 {
                        phase0_prob += p;
                    } else if g1 == 0 && g2 == 1 {
                        phase1_prob += p;
                    }
                });
            });

            phase[i] = phase1_prob / (phase0_prob + phase1_prob);
        }
    }

    println!("time = {} ms", (Instant::now() - start).as_millis());

    phase
}

// random initialization
fn gen_data(m: usize, n: usize) -> (Vec<Vec<u8>>, Vec<u8>) {
    use rand::Rng;
    let mut rng = rand::thread_rng();
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
        t[i] = 1; // 0, 1, or 2
    }
    (x, t)
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
