use crate::genotype_graph::GenotypeGraph;
use crate::{Genotype, Real, UInt, U8};
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};
use rand::distributions::{Distribution, WeightedIndex};
use rand::{rngs::ThreadRng, thread_rng, Rng};
use std::time::Instant;

#[cfg(feature = "leak-resist")]
use tp_fixedpoint::timing_shield::{TpEq, TpOrd};

pub fn hmm() {
    let mut rng = rand::thread_rng();

    let m = 20000; // num of variants
    let n = 1000; // num of samples
    let het_per_block = 3;
    let p = 1 << het_per_block;

    // hmm parameters
    let rprob = 0.05; // recombination
    let eprob = 0.01; // error

    let mut x = unsafe { Array2::<i8>::uninit((m, n)).assume_init() };
    let mut t = unsafe { Array1::<Genotype>::uninit(m).assume_init() }; // target

    for v in x.iter_mut() {
        *v = rng.gen_range(0..2);
    }

    // Simulate haplotypes from x then combine to get example target
    {
        let mut thap = Array2::<i8>::zeros((2, m));
        for hap in 0..2 {
            thap.row_mut(hap)
                .assign(&sample_from_hmm(x.view(), rprob, eprob, &mut thread_rng()));
        }
        for i in 0..m {
            #[cfg(feature = "leak-resist")]
            {
                t[i] = Genotype::protect(thap[[0, i]] + thap[[1, i]]);
            }

            #[cfg(not(feature = "leak-resist"))]
            {
                t[i] = thap[[0, i]] + thap[[1, i]];
            }
        }
    }

    // Geno graph for target
    let now = Instant::now();
    let mut geno_graph = GenotypeGraph::build(t.view(), het_per_block);
    println!(
        "Build genotype graph: {} ms",
        (Instant::now() - now).as_millis()
    );

    // TODO update ref
    #[cfg(feature = "leak-resist")]
    let x = {
        let mut _x = unsafe { Array2::<Genotype>::uninit((m, n)).assume_init() };
        for (a, &b) in _x.iter_mut().zip(x.iter()) {
            *a = Genotype::protect(b);
        }
        _x
    };

    let now = Instant::now();
    let bprob = backward_pass(x.view(), &geno_graph, rprob, eprob);
    let (phase_ind, _) = forward_sampling(
        bprob.view(),
        x.view(),
        &geno_graph,
        rprob,
        eprob,
        &mut rng,
        0,
    );
    let mut phased = get_haps_from_graph(&geno_graph, phase_ind.view());
    println!(
        "Burn-in iteration: {} ms",
        (Instant::now() - now).as_millis()
    );

    let now = Instant::now();
    //TODO update ref
    let bprob = backward_pass(x.view(), &geno_graph, rprob, eprob);
    let (phase_ind, tprob) = forward_sampling(
        bprob.view(),
        x.view(),
        &geno_graph,
        rprob,
        eprob,
        &mut rng,
        2,
    );

    phased = get_haps_from_graph(&geno_graph, phase_ind.view());
    geno_graph.prune(tprob.unwrap().view());
    println!(
        "Pruning iteration: {} ms",
        (Instant::now() - now).as_millis()
    );

    let num_main_iter = 3;
    //let mut tprob_agg = vec![vec![vec![0.0; p]; p]; m]; // to store aggregated transition probs
    let mut tprob_agg = unsafe { Array3::<Real>::uninit((m, p, p)).assume_init() }; // to store aggregated transition probs
    for i in 0..num_main_iter {
        let now = Instant::now();
        //update_ref_panel(&x, &phased);
        let bprob = backward_pass(x.view(), &geno_graph, rprob, eprob);
        let (phase_ind, tprob) = forward_sampling(
            bprob.view(),
            x.view(),
            &geno_graph,
            rprob,
            eprob,
            &mut rng,
            1,
        );
        phased = get_haps_from_graph(&geno_graph, phase_ind.view());

        // Add tprob to aggregator
        tprob_agg = &tprob_agg + &tprob.unwrap();
        println!(
            "Main iteration {}: {} ms",
            i,
            (Instant::now() - now).as_millis()
        );
    }

    let now = Instant::now();
    // Maximum a posteriori decoding using averaged trans probs
    let phase_ind = viterbi(tprob_agg.view());
    println!("Viterbi: {} ms", (Instant::now() - now).as_millis());
    phased = get_haps_from_graph(&geno_graph, phase_ind.view());
}

// TODO make this side channel resilient
fn sample(weights: &[f32], rng: &mut ThreadRng) -> usize {
    let dist = WeightedIndex::new(&*weights).unwrap();
    dist.sample(rng)
}

// TODO make this side channel resilient
fn sample_real(weights: ArrayView1<Real>, rng: &mut ThreadRng) -> UInt {
    #[cfg(feature = "leak-resist")]
    let weights = Array1::from_vec(
        weights
            .iter()
            .map(|v| v.leaky_into_f32())
            .collect::<Vec<_>>(),
    );

    #[cfg(not(feature = "leak-resist"))]
    let weights = weights.to_owned();

    let dist = WeightedIndex::new(weights.into_raw_vec());

    #[cfg(feature = "leak-resist")]
    {
        if dist.is_err() {
            return UInt::protect(0);
        } else {
            UInt::protect(dist.unwrap().sample(rng) as u32)
        }
    }

    #[cfg(not(feature = "leak-resist"))]
    {
        if dist.is_err() {
            return 0; 
        } else {
            dist.unwrap().sample(rng) as u32
        }
    }
}

// Utility function for simulating target sequences for testing purposes
fn sample_from_hmm(x: ArrayView2<i8>, rprob: f32, eprob: f32, rng: &mut ThreadRng) -> Array1<i8> {
    let m = x.nrows();
    let n = x.ncols();

    let mut hap = Array1::<i8>::zeros(m);

    let mut recvec = vec![0.0; 2];
    recvec[0] = 1.0 - rprob;
    recvec[1] = rprob;

    let mut errvec = vec![0.0; 2];
    errvec[0] = 1.0 - eprob;
    errvec[1] = eprob;

    let mut z = rng.gen_range(0..n);
    hap[0] = (x[[0, z]] + sample(errvec.as_slice(), rng) as i8) % 2;

    for i in 1..m {
        let rec_flag = sample(recvec.as_slice(), rng);
        if rec_flag == 1 {
            z = rng.gen_range(0..n);
        }
        hap[i] = (x[[i, z]] + sample(errvec.as_slice(), rng) as i8) % 2;
    }

    hap
}

fn emission_prob(x_geno: Genotype, t_geno: Genotype, error_prob: f32) -> Real {
    #[cfg(feature = "leak-resist")]
    {
        x_geno.tp_eq(&t_geno).select(
            Real::leaky_from_f32(1.0 - error_prob),
            Real::leaky_from_f32(error_prob),
        )
    }

    #[cfg(not(feature = "leak-resist"))]
    {
        if x_geno == t_geno {
            1.0 - error_prob
        } else {
            error_prob
        }
    }
}

fn transition_prob(prev_prob: Real, total_prob: Real, uniform_frac: f32, recomb_prob: f32) -> Real {
    #[cfg(feature = "leak-resist")]
    {
        prev_prob * Real::leaky_from_f32(1.0 - recomb_prob)
            + total_prob * Real::leaky_from_f32(recomb_prob * uniform_frac)
    }

    #[cfg(not(feature = "leak-resist"))]
    {
        prev_prob * (1.0 - recomb_prob) + total_prob * recomb_prob * uniform_frac
    }
}

fn backward_pass(
    x: ArrayView2<Genotype>,
    graph: &GenotypeGraph,
    rprob: f32,
    eprob: f32,
) -> Array3<Real> {
    let m = x.nrows();
    let n = x.ncols();
    let p = graph.p;

    let mut bprob = unsafe { Array3::<Real>::uninit((m, p, n)).assume_init() };

    // initialization (uniform over ref haplotypes + emission at last position)
    for h1 in 0..p {
        for h2 in 0..n {
            bprob[[m - 1, h1, h2]] = emission_prob(x[[m - 1, h2]], graph.graph[[m - 1, h1]], eprob);
        }
    }

    // backward pass
    for i in (0..m - 1).rev() {
        // i -> i+1 transition
        let mut h1sum = unsafe { Array1::<Real>::uninit(n).assume_init() };
        for h1 in 0..p {
            let h2sum = bprob.slice(s![i + 1, h1, ..]).iter().sum();
            for h2 in 0..n {
                bprob[[i, h1, h2]] =
                    transition_prob(bprob[[i + 1, h1, h2]], h2sum, 1.0 / (n as f32), rprob);
            }

            // Add to aggregate sum over h1
            for h2 in 0..n {
                if h1 == 0 {
                    h1sum[h2] = bprob[[i, h1, h2]];
                } else {
                    h1sum[h2] += bprob[[i, h1, h2]];
                }
            }
        }

        // If i+1 is block head, then replace bprob with its sum over h1
        // (between blocks, any transition between different values of h1 is possible)
        for h1 in 0..p {
            for h2 in 0..n {
                #[cfg(feature = "leak-resist")]
                {
                    bprob[[i, h1, h2]] =
                        graph.block_head[i + 1].select(h1sum[h2], bprob[[i, h1, h2]]);
                }
                #[cfg(not(feature = "leak-resist"))]
                {
                    if graph.block_head[i + 1] {
                        bprob[[i, h1, h2]] = h1sum[h2];
                    }
                }
            }
        }

        // emission at i
        for h1 in 0..p {
            for h2 in 0..n {
                bprob[[i, h1, h2]] *= emission_prob(x[[i, h2]], graph.graph[[i, h1]], eprob);
            }
        }

        // renormalize (TODO: replace with lazy normalization)
        let bsum: Real = bprob.slice(s![i, .., ..]).iter().sum();
        for h1 in 0..p {
            for h2 in 0..n {
                bprob[[i, h1, h2]] /= bsum;
            }
        }
    }
    bprob
}

// Only pairs of indices that add up to n (length of the weight vectors) are allowed
// Weight of a pair is the product of the two weights (joint probability)
fn constrained_paired_sample(
    weights1: ArrayView1<Real>,
    weights2: ArrayView1<Real>,
    rng: &mut ThreadRng,
) -> (UInt, UInt) {
    let n = weights1.len();
    let mut combined = unsafe { Array1::<Real>::uninit(n).assume_init() };
    for i in 0..n {
        combined[i] = weights1[i] * weights2[n - 1 - i];
    }

    // TODO fix this
    let ind1 = sample_real(combined.view(), rng);

    #[cfg(feature = "leak-resist")]
    {
        (ind1, UInt::protect(n as u32) - 1 - ind1)
    }

    #[cfg(not(feature = "leak-resist"))]
    {
        (ind1, n as u32 - 1 - ind1)
    }
}

fn inner_prod(v1: ArrayView1<Real>, v2: ArrayView1<Real>) -> Real {
    let prod = &v1 * &v2;
    prod.iter().sum()
}

// Forward sampling. Returns paired indices (diploid) into target genotype graph
// Optionally returns transition probabilities between adjacent positions in 2nd slot
// If not returning trans probs only compute forward probabilities needed for sampled states
//
// If trans_prob_flag = 0, do not return trans prob and run a focused forward pass
// If trans_prob_flag = 1, output p(x1), p(x2|x1), p(x3|x2) ... where xi denotes
// index into target genotype graph at position i
// If trans_prob_flag = 2, output p(x1), p(x1,x2), p(x2,x3) ... instead
fn forward_sampling(
    bprob: ArrayView3<Real>,
    x: ArrayView2<Genotype>,
    graph: &GenotypeGraph,
    rprob: f32,
    eprob: f32,
    rng: &mut ThreadRng,
    trans_prob_flag: usize,
) -> (Array2<U8>, Option<Array3<Real>>) {
    let m = x.nrows();
    let n = x.ncols();
    let p = graph.p;

    //let mut phase_ind = vec![vec![0 as u8; m]; 2];
    let mut phase_ind = unsafe { Array2::<U8>::uninit((2, m)).assume_init() };

    // p x p matrix at each pos i for transition between i-1 and i
    // Save belief over the first block in tprob[0][0]

    let mut firstprob = unsafe { Array1::<Real>::uninit(p).assume_init() };
    for h1 in 0..p {
        firstprob[h1] = bprob.slice(s![0, h1, ..]).iter().sum();
    }

    let mut tprob = None;
    if trans_prob_flag > 0 {
        //tprob = Some(vec![vec![vec![0.0 as f32; p]; p]; m]);
        tprob = Some(unsafe { Array3::<Real>::uninit((m, p, p)).assume_init() });
        for h1 in 0..p {
            tprob.as_mut().unwrap()[[0, 0, h1]] = firstprob[h1];
        }
    }

    // Sample first position
    let (ind1, ind2) = constrained_paired_sample(firstprob.view(), firstprob.view(), rng);
    #[cfg(feature = "leak-resist")]
    {
        phase_ind[[0, 0]] = ind1.as_u8();
        phase_ind[[1, 0]] = ind2.as_u8();
    }

    #[cfg(not(feature = "leak-resist"))]
    {
        phase_ind[[0, 0]] = ind1 as u8;
        phase_ind[[1, 0]] = ind2 as u8;
    }

    let psub = if trans_prob_flag == 0 { 2 } else { p };

    // Initialize forward prob
    let mut fprob = unsafe { Array2::<Real>::uninit((psub, n)).assume_init() };
    let mut fprob_next = unsafe { Array2::<Real>::uninit((psub, n)).assume_init() };

    // Emission at first position
    for h1 in 0..psub {
        for h2 in 0..n {
            let graph_ind = if trans_prob_flag == 0 {
                phase_ind[[h1, 0]]
            } else {
                #[cfg(feature = "leak-resist")]
                {
                    U8::protect(h1 as u8)
                }
                #[cfg(not(feature = "leak-resist"))]
                {
                    h1 as u8
                }
            };

            // TODO make oblivious
            #[cfg(feature = "leak-resist")]
            {
                fprob[[h1, h2]] = emission_prob(
                    x[[0, h2]],
                    graph.graph[[0, graph_ind.expose() as usize]],
                    eprob,
                );
            }

            #[cfg(not(feature = "leak-resist"))]
            {
                fprob[[h1, h2]] =
                    emission_prob(x[[0, h2]], graph.graph[[0, graph_ind as usize]], eprob);
            }
        }
    }

    for i in 1..m {
        // Combine fprob (from i-1) and bprob[i] to get transition probs
        let mut weights = unsafe { Array2::<Real>::uninit((psub, p)).assume_init() };

        for j in 0..psub {
            // Transition i-1 -> i
            let fsum = fprob.row(j).iter().sum();
            for h2 in 0..n {
                fprob_next[[j, h2]] =
                    transition_prob(fprob[[j, h2]], fsum, 1.0 / (n as f32), rprob);
            }

            // Multiply with backward prob and integrate to get transition prob from hap j to h1
            for h1 in 0..p {
                let iprod = inner_prod(fprob_next.row(j), bprob.slice(s![i, h1, ..]));

                // If not between blocks, set to identity matrix
                let ind = if trans_prob_flag > 0 {
                    #[cfg(feature = "leak-resist")]
                    {
                        U8::protect(j as u8)
                    }
                    #[cfg(not(feature = "leak-resist"))]
                    {
                        j as u8
                    }
                } else {
                    phase_ind[[j, i - 1]]
                };

                #[cfg(feature = "leak-resist")]
                {
                    weights[[j, h1]] = ind.tp_eq(&U8::protect(h1 as u8)).select(
                        graph.block_head[i].select(iprod, Real::leaky_from_f32(1.0)),
                        graph.block_head[i].select(iprod, Real::NAN),
                    );
                }

                #[cfg(not(feature = "leak-resist"))]
                {
                    if ind as usize == h1 {
                        weights[[j, h1]] = if graph.block_head[i] { iprod } else { 1.0 };
                    } else {
                        weights[[j, h1]] = if graph.block_head[i] { iprod } else { 0.0 };
                    }
                }
            }
        }

        #[cfg(feature = "leak-resist")]
        let (h1, h2) = if trans_prob_flag > 0 {
            (phase_ind[[0, i - 1]], phase_ind[[1, i - 1]])
        } else {
            (U8::protect(0), U8::protect(1))
        };

        #[cfg(not(feature = "leak-resist"))]
        let (h1, h2) = if trans_prob_flag > 0 {
            (phase_ind[[0, i - 1]], phase_ind[[1, i - 1]])
        } else {
            (0, 1)
        };

        // TODO fix this
        #[cfg(feature = "leak-resist")]
        let (ind1, ind2) = constrained_paired_sample(
            weights.row(h1.expose() as usize),
            weights.row(h2.expose() as usize),
            rng,
        );

        #[cfg(not(feature = "leak-resist"))]
        let (ind1, ind2) =
            constrained_paired_sample(weights.row(h1 as usize), weights.row(h2 as usize), rng);

        // If i is NOT block head, then just keep the previous indices
        #[cfg(feature = "leak-resist")]
        {
            phase_ind[[0, i]] = graph.block_head[i].select(ind1.as_u8(), phase_ind[[0, i - 1]]);
            phase_ind[[1, i]] = graph.block_head[i].select(ind2.as_u8(), phase_ind[[1, i - 1]]);
        }

        #[cfg(not(feature = "leak-resist"))]
        {
            phase_ind[[0, i]] = if graph.block_head[i] {
                ind1 as u8
            } else {
                phase_ind[[0, i - 1]]
            };

            phase_ind[[1, i]] = if graph.block_head[i] {
                ind2 as u8
            } else {
                phase_ind[[1, i - 1]]
            };
        }

        // Copy and renormalize
        if trans_prob_flag > 0 {
            if trans_prob_flag == 1 {
                for h1 in 0..psub {
                    let tsum: Real = weights.row(h1).iter().sum();
                    for h2 in 0..p {
                        tprob.as_mut().unwrap()[[i, h1, h2]] = weights[[h1, h2]] / tsum;
                    }
                }
            } else {
                let mut tsum: Real = weights.row(0).iter().sum();
                for h1 in 1..psub {
                    tsum += weights.row(h1).iter().sum::<Real>();
                }
                for h1 in 0..psub {
                    for h2 in 0..p {
                        tprob.as_mut().unwrap()[[i, h1, h2]] = weights[[h1, h2]] / tsum;
                    }
                }
            }
        }

        // Add emission at i (with the sampled haplotypes) to fprob_next
        for h1 in 0..psub {
            for h2 in 0..n {
                let graph_ind = if trans_prob_flag == 0 {
                    phase_ind[[h1, i]]
                } else {
                    #[cfg(feature = "leak-resist")]
                    {
                        U8::protect(h1 as u8)
                    }
                    #[cfg(not(feature = "leak-resist"))]
                    {
                        h1 as u8
                    }
                };

                //TODO fix this

                #[cfg(feature = "leak-resist")]
                {
                    fprob_next[[h1, h2]] *= emission_prob(
                        x[[i, h2]],
                        graph.graph[[i, graph_ind.expose() as usize]],
                        eprob,
                    );
                }
                #[cfg(not(feature = "leak-resist"))]
                {
                    fprob_next[[h1, h2]] *=
                        emission_prob(x[[i, h2]], graph.graph[[i, graph_ind as usize]], eprob);
                }
            }
        }

        // Update fprob
        for h1 in 0..psub {
            for h2 in 0..n {
                fprob[[h1, h2]] = fprob_next[[h1, h2]];
            }
        }

        // Renormalize (TODO: replace with lazy normalization)
        let mut fsum: Real = fprob.row(0).iter().sum();
        for h1 in 1..psub {
            fsum += fprob.row(h1).iter().sum::<Real>();
        }
        for h1 in 0..psub {
            for h2 in 0..n {
                fprob[[h1, h2]] /= fsum;
            }
        }
    }

    (phase_ind, tprob)
}

fn get_haps_from_graph(graph: &GenotypeGraph, phase_ind: ArrayView2<U8>) -> Array2<Genotype> {
    let m = graph.graph.nrows();

    let mut phased = unsafe { Array2::<Genotype>::uninit((2, m)).assume_init() };

    for hap in 0..2 {
        for i in 0..m {
            //TODO fix this
            #[cfg(feature = "leak-resist")]
            {
                phased[[hap, i]] = graph.graph[[i, phase_ind[[hap, i]].expose() as usize]];
            }

            #[cfg(not(feature = "leak-resist"))]
            {
                phased[[hap, i]] = graph.graph[[i, phase_ind[[hap, i]] as usize]];
            }
        }
    }
    phased
}

// Takes in p(x1), p(x2|x1), ..., p(xn|xn-1)
// Outputs most likely (paired) assignment using Viterbi algorithm
fn viterbi(tprob: ArrayView3<Real>) -> Array2<U8> {
    let m = tprob.shape()[0];
    let p = tprob.shape()[1];

    let mut backtrace = unsafe { Array2::<U8>::uninit((m, p)).assume_init() };
    let mut maxprob = unsafe { Array1::<Real>::uninit(p).assume_init() };
    let mut maxprob_next = unsafe { Array1::<Real>::uninit(p).assume_init() };
    for h1 in 0..p {
        maxprob[h1] = tprob[[0, 0, h1]] * tprob[[0, 0, p - 1 - h1]]; // p(x1), diploid prob
    }

    for i in 1..m {
        for h2 in 0..p {
            let mut max_val: Real = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
            let mut max_ind: U8 = unsafe { std::mem::MaybeUninit::uninit().assume_init() };

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

        // TODO: replace with lazy renormalization based on shifts
        let sum: Real = maxprob_next.iter().sum();
        for h2 in 0..p {
            maxprob[h2] = maxprob_next[h2] / sum;
        }
    }

    // Find max value in the final vector
    let mut max_val: Real = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
    let mut max_ind: U8 = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
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
