use crate::{tp_convert_to, tp_value, Bool, Genotype, Real, UInt, U8};
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};
use rand::Rng;
use std::time::Instant;

#[cfg(feature = "leak-resist")]
mod inner {
    use super::*;
    pub use tp_fixedpoint::timing_shield::{TpBool, TpCondSwap, TpEq, TpOrd};
    pub struct SortItem {
        pub dip: Real,
        pub i: U8,
        pub j: U8,
    }

    impl TpOrd for SortItem {
        fn tp_lt(&self, rhs: &Self) -> TpBool {
            self.dip.tp_lt(&rhs.dip)
        }

        fn tp_lt_eq(&self, rhs: &Self) -> TpBool {
            self.dip.tp_lt_eq(&rhs.dip)
        }

        fn tp_gt(&self, rhs: &Self) -> TpBool {
            self.dip.tp_gt(&rhs.dip)
        }

        fn tp_gt_eq(&self, rhs: &Self) -> TpBool {
            self.dip.tp_gt_eq(&rhs.dip)
        }
    }

    impl TpCondSwap for SortItem {
        fn tp_cond_swap(cond: TpBool, a: &mut Self, b: &mut Self) {
            Real::tp_cond_swap(cond, &mut a.dip, &mut b.dip);
            U8::tp_cond_swap(cond, &mut a.i, &mut b.i);
            U8::tp_cond_swap(cond, &mut a.j, &mut b.j);
        }
    }
}

#[cfg(feature = "leak-resist")]
use inner::*;

pub struct GenotypeGraph {
    pub graph: Array2<Genotype>,
    pub block_head: Array1<Bool>,
    pub p: usize,
}

pub const HET_PER_BLOCK: usize = 3;

impl GenotypeGraph {
    pub fn build(t: ArrayView1<Genotype>) -> Self {
        let m = t.len();
        let p = 1 << HET_PER_BLOCK;
        let mut graph = unsafe { Array2::<Genotype>::uninit((m, p)).assume_init() };
        let mut block_head = unsafe { Array1::<Bool>::uninit(m).assume_init() };

        let mut cur_het_count = tp_value!(HET_PER_BLOCK, u32);

        for i in 0..m {
            #[cfg(feature = "leak-resist")]
            {
                let head_flag = cur_het_count.tp_eq(&(HET_PER_BLOCK as u32));
                block_head[i] = head_flag;
                cur_het_count = head_flag.select(UInt::protect(0), cur_het_count);

                let het_flag = t[i].tp_eq(&1);
                cur_het_count = het_flag.select(cur_het_count + 1, cur_het_count);

                let start_geno = t[i]
                    .tp_lt(&2)
                    .select(Genotype::protect(0), Genotype::protect(1));
                let step_size = het_flag.select(
                    oram_sgx::utils::tp_u32_shl(
                        UInt::protect(1),
                        cur_het_count - 1,
                        HET_PER_BLOCK as u32,
                    ),
                    UInt::protect(p as u32),
                );
                let mut cur_geno = start_geno;
                for j in 0..p {
                    graph[[i, j]] = cur_geno;
                    let (_, modulo) = oram_sgx::utils::tp_u32_div(
                        UInt::protect(j as u32 + 1),
                        step_size,
                        p as u32,
                    );
                    cur_geno = modulo.tp_eq(&0).select(cur_geno + 1 & 1, cur_geno);
                }
            }

            #[cfg(not(feature = "leak-resist"))]
            {
                let head_flag = cur_het_count == HET_PER_BLOCK as u32;
                block_head[i] = head_flag;
                // Reset if at block head
                if head_flag {
                    cur_het_count = 0;
                }
                let het_flag = t[i] == 1;
                if het_flag {
                    cur_het_count += 1;
                }

                let start_geno = if t[i] < 2 { 0 } else { 1 };

                let step_size = if het_flag {
                    1 << (cur_het_count - 1)
                } else {
                    p
                };
                let mut cur_geno = start_geno;
                for j in 0..p {
                    graph[[i, j]] = cur_geno;
                    if ((j + 1) % step_size) == 0 {
                        cur_geno = (cur_geno + 1) % 2;
                    }
                }
            }
        }

        block_head[0] = tp_value!(false, bool);

        Self {
            graph,
            block_head,
            p,
        }
    }

    // TODO: limit merges to maximum of MAX_AMBIGUOUS het sites within a block
    pub fn prune(&mut self, tprob: ArrayView3<Real>) {
        let now = Instant::now();
        const MCMC_PRUNE_PROB_THRES: f32 = 0.999; // "mcmc-prune" parameter in ShapeIt4
        const MAX_AMBIGUOUS: usize = 22; // "MAX_AMB" constant in ShapeIt4 (utils/otools.h)

        let m = self.graph.nrows();
        let p = self.p;

        // Backward pass to identify where to merge adjacent blocks and the new haps
        let mut ind1_cache = unsafe { Array2::<U8>::uninit((m, p)).assume_init() };
        let mut ind2_cache = unsafe { Array2::<U8>::uninit((m, p)).assume_init() };
        let mut merge_head = vec![tp_value!(false, bool); m];
        let mut merge_flag = tp_value!(false, bool);
        let mut new_merge_flag;

        for i in (0..m - 1).rev() {
            let (ind1, ind2, prob) = select_top_k(tprob.slice(s![i, .., ..]), p);
            // If merge flag set then carry over
            for j in 0..p {
                #[cfg(feature = "leak-resist")]
                {
                    ind1_cache[[i, j]] = merge_flag.select(ind1_cache[[i + 1, j]], ind1[j]);
                    ind2_cache[[i, j]] = merge_flag.select(ind2_cache[[i + 1, j]], ind2[j]);
                }

                #[cfg(not(feature = "leak-resist"))]
                {
                    ind1_cache[[i, j]] = if merge_flag {
                        ind1_cache[[i + 1, j]]
                    } else {
                        ind1[j]
                    };
                    ind2_cache[[i, j]] = if merge_flag {
                        ind2_cache[[i + 1, j]]
                    } else {
                        ind2[j]
                    };
                }
            }

            #[cfg(feature = "leak-resist")]
            {
                // If at head and merge is on, it is the merge head
                merge_head[i] = (self.block_head[i] | (i == 0)) & merge_flag;

                // If merge is not on, we're at a head and prob over threshold, start a new merge
                new_merge_flag = self.block_head[i]
                    & prob.tp_gt(&Real::leaky_from_f32(MCMC_PRUNE_PROB_THRES))
                    & !merge_flag;

                // If at head then merge_flag is set to new_merge_flag, otherwise carry over
                merge_flag = self.block_head[i].select(new_merge_flag, merge_flag);
            }

            #[cfg(not(feature = "leak-resist"))]
            {
                // If at head and merge is on, it is the merge head
                merge_head[i] = (i == 0 || self.block_head[i]) && merge_flag;

                // If merge is not on, we're at a head and prob over threshold, start a new merge
                new_merge_flag = self.block_head[i] && prob > MCMC_PRUNE_PROB_THRES && !merge_flag;

                // If at head then merge_flag is set to new_merge_flag, otherwise carry over
                if self.block_head[i] {
                    merge_flag = new_merge_flag;
                }
            }
        }

        // Forward pass to update the graph and block_head
        let mut new_geno = unsafe { Array1::<Genotype>::uninit(p).assume_init() };
        let mut ind1 = unsafe { Array1::<U8>::uninit(p).assume_init() };
        let mut ind2 = unsafe { Array1::<U8>::uninit(p).assume_init() };
        let mut ind = unsafe { Array1::<U8>::uninit(p).assume_init() };
        let mut block_counter = tp_value!(0, i8);

        for i in 0..m {
            #[cfg(feature = "leak-resist")]
            {
                block_counter = merge_head[i].select(tp_value!(2, i8), block_counter);
                block_counter = (!merge_head[i] & self.block_head[i]).select(
                    (block_counter)
                        .tp_gt(&1)
                        .select(block_counter - 1, tp_value!(0, i8)),
                    block_counter,
                );
                for j in 0..p {
                    // If at merge head copy cache over
                    ind1[j] = merge_head[i].select(ind1_cache[[i, j]], ind1[j]);
                    ind2[j] = merge_head[i].select(ind2_cache[[i, j]], ind2[j]);

                    // If block_counter hits zero change back to original indicies
                    ind1[j] = block_counter
                        .tp_eq(&0)
                        .select(U8::protect(j as u8), ind1[j]);
                    ind2[j] = block_counter
                        .tp_eq(&0)
                        .select(U8::protect(j as u8), ind2[j]);

                    // Use ind1 for first block, ind2 for second block
                    ind[j] = block_counter.tp_eq(&2).select(ind1[j], ind2[j]);
                }

                // Update graph haplotypes TODO: use ORAM to hide ind
                // TODO fix this
                for j in 0..p {
                    new_geno[j] = self.graph[[i, ind[j].expose() as usize]];
                }

                for j in 0..p {
                    self.graph[[i, j]] = new_geno[j];
                }

                // Erase block_head flag between the two blocks being merged
                self.block_head[i] = (self.block_head[i] & block_counter.tp_eq(&1))
                    .select(Bool::protect(false), self.block_head[i]);
            }

            #[cfg(not(feature = "leak-resist"))]
            {
                // At merge head set block counter to 2 (will update the next two blocks)
                if merge_head[i] {
                    block_counter = 2;
                }
                // Every time we see a block head that is not a merge head, decrement block counter
                if !merge_head[i] && self.block_head[i] {
                    block_counter = 0.max((block_counter as i8) - 1);
                }

                for j in 0..p {
                    // If at merge head copy cache over
                    if merge_head[i] {
                        ind1[j] = ind1_cache[[i, j]];
                        ind2[j] = ind2_cache[[i, j]];
                    }

                    // If block_counter hits zero change back to original indicies
                    if block_counter == 0 {
                        ind1[j] = j as u8;
                        ind2[j] = j as u8;
                    }

                    // Use ind1 for first block, ind2 for second block
                    if block_counter == 2 {
                        ind[j] = ind1[j];
                    } else {
                        ind[j] = ind2[j];
                    }
                }

                // Update graph haplotypes TODO: use ORAM to hide ind
                for j in 0..p {
                    new_geno[j] = self.graph[[i, ind[j] as usize]];
                }
                for j in 0..p {
                    self.graph[[i, j]] = new_geno[j];
                }

                // Erase block_head flag between the two blocks being merged
                if self.block_head[i] && block_counter == 1 {
                    self.block_head[i] = false;
                }
            }
        }
        println!(
            "Genotype graph pruning: {} ms",
            (Instant::now() - now).as_millis()
        );
    }

    // Forward sampling. Returns paired indices (diploid) into target genotype graph
    // Optionally returns transition probabilities between adjacent positions in 2nd slot
    // If not returning trans probs only compute forward probabilities needed for sampled states
    //
    // If trans_prob_flag = 0, do not return trans prob and run a focused forward pass
    // If trans_prob_flag = 1, output p(x1), p(x2|x1), p(x3|x2) ... where xi denotes
    // index into target genotype graph at position i
    // If trans_prob_flag = 2, output p(x1), p(x1,x2), p(x2,x3) ... instead
    pub fn forward_sampling(
        &self,
        x: ArrayView2<Genotype>,
        rprob: f32,
        eprob: f32,
        rng: &mut impl Rng,
        trans_prob_flag: usize,
    ) -> (Array2<U8>, Option<Array3<Real>>) {
        let m = x.nrows();
        let n = x.ncols();
        let p = self.p;

        println!("== Backward pass ==");
        let now = Instant::now();

        let bprob = self.backward_pass(x, rprob, eprob);

        println!("Backward pass: {} ms", (Instant::now() - now).as_millis());
        println!("");

        println!("== Forward sampling ==");
        let mut sum_time = vec![0; 6];
        let now_all = Instant::now();

        let mut phase_ind = unsafe { Array2::<U8>::uninit((2, m)).assume_init() };

        // p x p matrix at each pos i for transition between i-1 and i
        // Save belief over the first block in tprob[0][0]
        let now = Instant::now();
        let firstprob = Array1::from_shape_fn(p, |i| bprob.slice(s![0, i, ..]).iter().sum());
        sum_time[0] = (Instant::now() - now).as_nanos();

        let mut tprob = None;
        if trans_prob_flag > 0 {
            tprob = Some(unsafe { Array3::<Real>::uninit((m, p, p)).assume_init() });
            tprob
                .as_mut()
                .unwrap()
                .slice_mut(s![0, 0, ..])
                .assign(&firstprob);
        }

        // Sample first position
        let (ind1, ind2) = constrained_paired_sample(firstprob.view(), firstprob.view(), rng);
        phase_ind[[0, 0]] = tp_convert_to!(ind1, u8);
        phase_ind[[1, 0]] = tp_convert_to!(ind2, u8);

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
                    tp_value!(h1, u8)
                };

                // TODO make oblivious
                #[cfg(feature = "leak-resist")]
                {
                    fprob[[h1, h2]] = emission_prob(
                        x[[0, h2]],
                        self.graph[[0, graph_ind.expose() as usize]],
                        eprob,
                    );
                }

                #[cfg(not(feature = "leak-resist"))]
                {
                    fprob[[h1, h2]] =
                        emission_prob(x[[0, h2]], self.graph[[0, graph_ind as usize]], eprob);
                }
            }
        }

        for i in 1..m {
            // Combine fprob (from i-1) and bprob[i] to get transition probs
            let mut weights = unsafe { Array2::<Real>::uninit((psub, p)).assume_init() };

            for j in 0..psub {
                let now = Instant::now();
                // Transition i-1 -> i
                let fsum = fprob.row(j).iter().sum();
                sum_time[1] += (Instant::now() - now).as_nanos();
                for h2 in 0..n {
                    fprob_next[[j, h2]] =
                        transition_prob(fprob[[j, h2]], fsum, 1.0 / (n as f32), rprob);
                }

                // Multiply with backward prob and integrate to get transition prob from hap j to h1
                for h1 in 0..p {
                    let now = Instant::now();
                    let iprod = inner_prod(fprob_next.row(j), bprob.slice(s![i, h1, ..]));
                    sum_time[2] += (Instant::now() - now).as_nanos();

                    // If not between blocks, set to identity matrix
                    let ind = if trans_prob_flag > 0 {
                        tp_value!(j, u8)
                    } else {
                        phase_ind[[j, i - 1]]
                    };

                    #[cfg(feature = "leak-resist")]
                    {
                        weights[[j, h1]] = ind.tp_eq(&U8::protect(h1 as u8)).select(
                            self.block_head[i].select(iprod, Real::leaky_from_f32(1.0)),
                            self.block_head[i].select(iprod, Real::NAN),
                        );
                    }

                    #[cfg(not(feature = "leak-resist"))]
                    {
                        weights[[j, h1]] = if ind as usize == h1 {
                            if self.block_head[i] {
                                iprod
                            } else {
                                1.0
                            }
                        } else {
                            if self.block_head[i] {
                                iprod
                            } else {
                                0.0
                            }
                        }
                    }
                }
            }

            let (h1, h2) = if trans_prob_flag > 0 {
                (phase_ind[[0, i - 1]], phase_ind[[1, i - 1]])
            } else {
                (tp_value!(0, u8), tp_value!(1, u8))
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
                phase_ind[[0, i]] = self.block_head[i].select(ind1.as_u8(), phase_ind[[0, i - 1]]);
                phase_ind[[1, i]] = self.block_head[i].select(ind2.as_u8(), phase_ind[[1, i - 1]]);
            }

            #[cfg(not(feature = "leak-resist"))]
            {
                phase_ind[[0, i]] = if self.block_head[i] {
                    ind1 as u8
                } else {
                    phase_ind[[0, i - 1]]
                };

                phase_ind[[1, i]] = if self.block_head[i] {
                    ind2 as u8
                } else {
                    phase_ind[[1, i - 1]]
                };
            }

            // Copy and renormalize
            if trans_prob_flag > 0 {
                if trans_prob_flag == 1 {
                    for h1 in 0..psub {
                        let now = Instant::now();
                        #[cfg(feature = "leak-resist")]
                        let tsum = {
                            let mut buf = weights.row(h1).as_slice().unwrap().to_vec();
                            Real::checked_sum_in_place(&mut buf)
                        };

                        #[cfg(not(feature = "leak-resist"))]
                        let tsum: Real = weights.row(h1).iter().sum();

                        sum_time[3] += (Instant::now() - now).as_nanos();
                        tprob
                            .as_mut()
                            .unwrap()
                            .slice_mut(s![i, h1, ..])
                            .assign(&(&weights.slice(s![h1, ..]) / tsum))
                    }
                } else {
                    let now = Instant::now();

                    #[cfg(feature = "leak-resist")]
                    let tsum = {
                        let mut buf = weights.as_slice().unwrap().to_vec();
                        Real::checked_sum_in_place(&mut buf)
                    };

                    #[cfg(not(feature = "leak-resist"))]
                    let tsum = weights.iter().sum::<Real>();

                    sum_time[4] += (Instant::now() - now).as_nanos();
                    tprob
                        .as_mut()
                        .unwrap()
                        .slice_mut(s![i, .., ..])
                        .assign(&(&weights.slice(s![.., ..]) / tsum));
                }
            }

            // Add emission at i (with the sampled haplotypes) to fprob_next
            for h1 in 0..psub {
                for h2 in 0..n {
                    let graph_ind = if trans_prob_flag == 0 {
                        phase_ind[[h1, i]]
                    } else {
                        tp_value!(h1, u8)
                    };

                    //TODO fix this
                    #[cfg(feature = "leak-resist")]
                    {
                        fprob_next[[h1, h2]] *= emission_prob(
                            x[[i, h2]],
                            self.graph[[i, graph_ind.expose() as usize]],
                            eprob,
                        );
                    }
                    #[cfg(not(feature = "leak-resist"))]
                    {
                        fprob_next[[h1, h2]] *=
                            emission_prob(x[[i, h2]], self.graph[[i, graph_ind as usize]], eprob);
                    }
                }
            }

            // Update fprob
            fprob.assign(&fprob_next);

            // Renormalize (TODO: replace with lazy normalization)
            let now = Instant::now();
            let fsum = fprob.iter().sum::<Real>();
            sum_time[5] += (Instant::now() - now).as_nanos();
            fprob /= fsum;
        }

        for (i, t) in sum_time.iter().enumerate() {
            println!("Summation {}: {} ms", i, t / 1000000);
        }
        println!(
            "Forward sampling: {} ms",
            (Instant::now() - now_all).as_millis()
        );
        println!("");

        (phase_ind, tprob)
    }

    pub fn get_haps(&self, phase_ind: ArrayView2<U8>) -> Array2<Genotype> {
        let now = Instant::now();
        let m = self.graph.nrows();

        let mut phased = unsafe { Array2::<Genotype>::uninit((2, m)).assume_init() };

        for hap in 0..2 {
            for i in 0..m {
                //TODO: fix this
                #[cfg(feature = "leak-resist")]
                {
                    phased[[hap, i]] = self.graph[[i, phase_ind[[hap, i]].expose() as usize]];
                }

                #[cfg(not(feature = "leak-resist"))]
                {
                    phased[[hap, i]] = self.graph[[i, phase_ind[[hap, i]] as usize]];
                }
            }
        }
        println!("Get haplotypes: {} ms", (Instant::now() - now).as_millis());
        phased
    }

    fn backward_pass(&self, x: ArrayView2<Genotype>, rprob: f32, eprob: f32) -> Array3<Real> {
        let mut sum_time = 0;

        let m = x.nrows();
        let n = x.ncols();
        let p = self.p;

        let mut bprob = unsafe { Array3::<Real>::uninit((m, p, n)).assume_init() };

        // initialization (uniform over ref haplotypes + emission at last position)
        for h1 in 0..p {
            for h2 in 0..n {
                bprob[[m - 1, h1, h2]] =
                    emission_prob(x[[m - 1, h2]], self.graph[[m - 1, h1]], eprob);
            }
        }

        // backward pass
        for i in (0..m - 1).rev() {
            // i -> i+1 transition
            for h1 in 0..p {
                let now = Instant::now();
                let h2sum = bprob.slice(s![i + 1, h1, ..]).iter().sum();
                sum_time += (Instant::now() - now).as_nanos();

                for h2 in 0..n {
                    bprob[[i, h1, h2]] =
                        transition_prob(bprob[[i + 1, h1, h2]], h2sum, 1.0 / (n as f32), rprob);
                }
            }

            // Add to aggregate sum over h1
            let now = Instant::now();
            let h1sum = Array1::from_shape_fn(n, |j| bprob.slice(s![i, .., j]).iter().sum());
            sum_time += (Instant::now() - now).as_nanos();

            // If i+1 is block head, then replace bprob with its sum over h1
            // (between blocks, any transition between different values of h1 is possible)
            for h1 in 0..p {
                for h2 in 0..n {
                    #[cfg(feature = "leak-resist")]
                    {
                        bprob[[i, h1, h2]] =
                            self.block_head[i + 1].select(h1sum[h2], bprob[[i, h1, h2]]);
                    }
                    #[cfg(not(feature = "leak-resist"))]
                    {
                        if self.block_head[i + 1] {
                            bprob[[i, h1, h2]] = h1sum[h2];
                        }
                    }
                }
            }

            // emission at i
            for h1 in 0..p {
                for h2 in 0..n {
                    bprob[[i, h1, h2]] *= emission_prob(x[[i, h2]], self.graph[[i, h1]], eprob);
                }
            }

            let now = Instant::now();
            // renormalize (TODO: replace with lazy normalization)
            let bsum: Real = bprob.slice(s![i, .., ..]).iter().sum();
            sum_time += (Instant::now() - now).as_nanos();

            bprob
                .slice_mut(s![i, .., ..])
                .iter_mut()
                .for_each(|x| *x /= bsum);
        }
        println!("Summation: {} ms", sum_time / 1000000);
        bprob
    }
}

// Takes a p x p matrix M and returns a list of 0-based
// index pairs (i,j) corresponding to top K largest
// elements according to M(i,j) * M(p-1-i,p-1-j)
// To avoid duplicates, restrict i <= p/2
// Also output associated probability mass
fn select_top_k(tab: ArrayView2<Real>, k: usize) -> (Array1<U8>, Array1<U8>, Real) {
    let p = tab.nrows();
    let n = p * p / 2;
    let mut elems = Vec::with_capacity(n);
    for i in 0..p / 2 {
        for j in 0..p {
            let dip = tab[[i, j]] * tab[[p - 1 - i, p - 1 - j]];
            #[cfg(feature = "leak-resist")]
            {
                elems.push(SortItem {
                    dip,
                    i: U8::protect(i as u8),
                    j: U8::protect(j as u8),
                });
            }

            #[cfg(not(feature = "leak-resist"))]
            {
                elems.push((dip, i as u8, j as u8));
            }
        }
    }
    #[cfg(feature = "leak-resist")]
    let tot_sum: Real = elems.iter().map(|v| v.dip).sum();

    #[cfg(not(feature = "leak-resist"))]
    let tot_sum: Real = elems.iter().map(|v| v.0).sum();

    // Descending sort
    #[cfg(feature = "leak-resist")]
    {
        oram_sgx::BiotonicSort::sort(&mut elems[..], false);
    }

    #[cfg(not(feature = "leak-resist"))]
    {
        elems.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
    }

    let mut ind1 = unsafe { Array1::<U8>::uninit(k).assume_init() };
    let mut ind2 = unsafe { Array1::<U8>::uninit(k).assume_init() };

    #[cfg(feature = "leak-resist")]
    let mut sum: Real = elems.iter().take(k / 2).map(|v| v.dip).sum();

    #[cfg(not(feature = "leak-resist"))]
    let mut sum: Real = elems.iter().take(k / 2).map(|v| v.0).sum();

    sum /= tot_sum;
    for i in 0..k / 2 {
        // Choose top k/2 then fill in the rest by inverting
        #[cfg(feature = "leak-resist")]
        {
            ind1[i] = elems[i].i;
            ind2[i] = elems[i].j;
            ind1[k - 1 - i] = (p as u8) - 1 - elems[i].i;
            ind2[k - 1 - i] = (p as u8) - 1 - elems[i].j;
        }
        #[cfg(not(feature = "leak-resist"))]
        {
            ind1[i] = elems[i].1;
            ind2[i] = elems[i].2;
            ind1[k - 1 - i] = (p as u8) - 1 - elems[i].1;
            ind2[k - 1 - i] = (p as u8) - 1 - elems[i].2;
        }
    }
    (ind1, ind2, sum)
}

// Only pairs of indices that add up to n (length of the weight vectors) are allowed
// Weight of a pair is the product of the two weights (joint probability)
fn constrained_paired_sample(
    weights1: ArrayView1<Real>,
    weights2: ArrayView1<Real>,
    rng: &mut impl Rng,
) -> (UInt, UInt) {
    let n = weights1.len();
    let mut combined = unsafe { Array1::<Real>::uninit(n).assume_init() };
    for i in 0..n {
        #[cfg(feature = "leak-resist")]
        {
            combined[i] = (weights1[i].is_nan() | weights2[n - 1 - i].is_nan())
                .select(Real::NAN, weights1[i] * weights2[n - 1 - i]);
        }

        #[cfg(not(feature = "leak-resist"))]
        {
            combined[i] = weights1[i] * weights2[n - 1 - i];
        }
    }

    let ind1 = weighted_sample(combined.view(), rng);

    (ind1, tp_value!(n, u32) - 1 - ind1)
}

fn weighted_sample(weights: ArrayView1<Real>, rng: &mut impl Rng) -> UInt {
    let mut total_weight = weights[0];
    let mut cumulative_weights: Vec<Real> = Vec::with_capacity(weights.len());
    cumulative_weights.push(total_weight);
    for &w in weights.iter().skip(1) {
        #[cfg(feature = "leak-resist")]
        {
            total_weight = w.is_nan().select(
                total_weight,
                total_weight.is_nan().select(w, total_weight + w),
            );
        }
        #[cfg(not(feature = "leak-resist"))]
        {
            total_weight += w;
        }
        cumulative_weights.push(total_weight);
    }

    #[cfg(feature = "leak-resist")]
    {
        let chosen_weight = Real::leaky_from_f32(rng.gen_range(0.0..1.0)) * total_weight;
        let mut index = UInt::protect(0);
        let mut done = Bool::protect(false);
        for w in cumulative_weights {
            done = (w.tp_gt(&chosen_weight) & !w.is_nan()).select(Bool::protect(true), done);
            index = (!done).select(index + 1, index);
        }
        index
    }

    #[cfg(not(feature = "leak-resist"))]
    {
        let chosen_weight = rng.gen_range(0.0..1.0) * total_weight;
        use std::cmp::Ordering;
        cumulative_weights
            .binary_search_by(|w| {
                if *w <= chosen_weight {
                    Ordering::Less
                } else {
                    Ordering::Greater
                }
            })
            .unwrap_err() as u32
    }
}

#[inline]
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

#[inline]
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

fn inner_prod(v1: ArrayView1<Real>, v2: ArrayView1<Real>) -> Real {
    let prod = &v1 * &v2;
    prod.iter().sum()
}

#[cfg(test)]
mod ref_algs {
    use rand::distributions::{Distribution, WeightedIndex};
    use rand::Rng;

    // First return object encodes the graph (2^het_per_block by M matrix)
    // Second return object encodes a boolean vector indicating block starting positions
    pub fn construct_geno_graph(t: &[u8], het_per_block: i32) -> (Vec<Vec<u8>>, Vec<bool>) {
        let m = t.len();
        let p = 1 << het_per_block as usize;

        let mut graph = vec![vec![0 as u8; p]; m];
        let mut block_head = vec![false; m];

        let mut cur_het_count = het_per_block as usize;
        for i in 0..m {
            let head_flag = cur_het_count == het_per_block as usize;
            block_head[i] = head_flag;

            // Reset if at block head
            cur_het_count = mutex2(head_flag, 0, cur_het_count);

            let het_flag = t[i] == 1;
            cur_het_count = mutex2(het_flag, cur_het_count + 1, cur_het_count);

            let start_geno = mutex2(t[i] < 2, 0, 1) as u8;
            let step_size = mutex2(het_flag, (1 << cur_het_count) >> 1, p) as usize;

            let mut cur_geno = start_geno;
            for j in 0..p {
                graph[i][j] = cur_geno;
                cur_geno = mutex2u8(((j + 1) % step_size) == 0, (cur_geno + 1) % 2, cur_geno);
            }
        }

        block_head[0] = false; // Skip first position
        (graph, block_head)
    }

    // Takes a p x p matrix M and returns a list of 0-based
    // index pairs (i,j) corresponding to top K largest
    // elements according to M(i,j) * M(p-1-i,p-1-j)
    // To avoid duplicates, restrict i <= p/2
    // Also output associated probability mass
    pub fn select_top_k(tab: &Vec<Vec<f32>>, k: usize) -> (Vec<u8>, Vec<u8>, f32) {
        let p = tab.len();
        let n = p * p / 2;
        let mut elems = vec![(0.0, 0, 0); n];
        let mut ind = 0;
        let mut tot_sum = 0.0;
        for i in 0..p / 2 {
            for j in 0..p {
                let dip = tab[i][j] * tab[p - 1 - i][p - 1 - j];
                elems[ind] = (dip, i as u8, j as u8);
                tot_sum += dip;
                ind += 1;
            }
        }

        // Descending sort
        elems.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());

        let mut ind1 = vec![0 as u8; k];
        let mut ind2 = vec![0 as u8; k];
        let mut sum = 0.0;
        for i in 0..k / 2 {
            // Choose top k/2 then fill in the rest by inverting
            sum += elems[i].0;
            ind1[i] = elems[i].1;
            ind2[i] = elems[i].2;
            ind1[k - 1 - i] = (p as u8) - 1 - elems[i].1;
            ind2[k - 1 - i] = (p as u8) - 1 - elems[i].2;
        }
        sum /= tot_sum;

        (ind1, ind2, sum)
    }

    pub fn prune_graph(
        graph: &mut Vec<Vec<u8>>,
        block_head: &mut Vec<bool>,
        tprob: &Vec<Vec<Vec<f32>>>,
    ) {
        const MCMC_PRUNE_PROB_THRES: f32 = 0.999; // "mcmc-prune" parameter in ShapeIt4
                                                  //let MAX_AMBIGUOUS = 22; // "MAX_AMB" constant in ShapeIt4 (utils/otools.h)

        let m = graph.len();
        let p = graph[0].len();

        // Backward pass to identify where to merge adjacent blocks and the new haps
        let mut ind1_cache = vec![vec![0 as u8; p]; m];
        let mut ind2_cache = vec![vec![0 as u8; p]; m];
        let mut merge_head = vec![false; m];
        let mut merge_flag = false;
        let mut new_merge_flag;
        for i in (0..m - 1).rev() {
            let (ind1, ind2, prob) = select_top_k(&tprob[i], p);

            // If merge flag set then carry over
            for j in 0..p {
                ind1_cache[i][j] = mutex2u8(merge_flag, ind1_cache[i + 1][j], ind1[j]);
                ind2_cache[i][j] = mutex2u8(merge_flag, ind2_cache[i + 1][j], ind2[j]);
            }

            // If at head and merge is on, it is the merge head
            merge_head[i] = (i == 0 || block_head[i]) && merge_flag;

            // If merge is not on, we're at a head and prob over threshold, start a new merge
            new_merge_flag = block_head[i] && prob > MCMC_PRUNE_PROB_THRES && !merge_flag;

            // If at head then merge_flag is set to new_merge_flag, otherwise carry over
            merge_flag = mutex2bool(block_head[i], new_merge_flag, merge_flag);
        }
    }

    pub fn backward_pass(
        x: &Vec<Vec<u8>>,
        graph: &Vec<Vec<u8>>,
        block_head: &Vec<bool>,
        rprob: f32,
        eprob: f32,
    ) -> Vec<Vec<Vec<f32>>> {
        let m = x.len();
        let n = x[0].len();
        let p = graph[0].len();

        let mut bprob = vec![vec![vec![0.0 as f32; n]; p]; m]; // backward prob

        // initialization (uniform over ref haplotypes + emission at last position)
        for h1 in 0..p {
            for h2 in 0..n {
                bprob[m - 1][h1][h2] = emission_prob(x[m - 1][h2], graph[m - 1][h1], eprob);
            }
        }

        // backward pass
        for i in (0..m - 1).rev() {
            // i -> i+1 transition
            let mut h1sum = vec![0.0; n];
            for h1 in 0..p {
                let h2sum = sum_vec(bprob[i + 1][h1].as_slice());
                for h2 in 0..n {
                    bprob[i][h1][h2] =
                        transition_prob(bprob[i + 1][h1][h2], h2sum, 1.0 / (n as f32), rprob);
                }

                // Add to aggregate sum over h1
                for h2 in 0..n {
                    h1sum[h2] += bprob[i][h1][h2];
                }
            }

            // If i+1 is block head, then replace bprob with its sum over h1
            // (between blocks, any transition between different values of h1 is possible)
            for h1 in 0..p {
                for h2 in 0..n {
                    bprob[i][h1][h2] = mutex2f32(block_head[i + 1], h1sum[h2], bprob[i][h1][h2]);
                }
            }

            // emission at i
            for h1 in 0..p {
                for h2 in 0..n {
                    bprob[i][h1][h2] *= emission_prob(x[i][h2], graph[i][h1], eprob);
                }
            }

            // renormalize (TODO: replace with lazy normalization)
            let mut bsum = 0.0;
            for h1 in 0..p {
                bsum += sum_vec(bprob[i][h1].as_slice());
            }
            for h1 in 0..p {
                for h2 in 0..n {
                    bprob[i][h1][h2] /= bsum;
                }
            }
        }
        bprob
    }

    // Forward sampling. Returns paired indices (diploid) into target genotype graph
    // Optionally returns transition probabilities between adjacent positions in 2nd slot
    // If not returning trans probs only compute forward probabilities needed for sampled states
    //
    // If trans_prob_flag = 0, do not return trans prob and run a focused forward pass
    // If trans_prob_flag = 1, output p(x1), p(x2|x1), p(x3|x2) ... where xi denotes
    // index into target genotype graph at position i
    // If trans_prob_flag = 2, output p(x1), p(x1,x2), p(x2,x3) ... instead
    pub fn forward_sampling(
        bprob: &Vec<Vec<Vec<f32>>>,
        x: &Vec<Vec<u8>>,
        graph: &Vec<Vec<u8>>,
        block_head: &Vec<bool>,
        rprob: f32,
        eprob: f32,
        rng: &mut impl Rng,
        trans_prob_flag: usize,
    ) -> (Vec<Vec<u8>>, Vec<Vec<Vec<f32>>>) {
        let m = x.len();
        let n = x[0].len();
        let p = graph[0].len();

        let mut phase_ind = vec![vec![0 as u8; m]; 2];

        // p x p matrix at each pos i for transition between i-1 and i
        // Save belief over the first block in tprob[0][0]

        let mut firstprob = vec![0.0 as f32; p];
        for h1 in 0..p {
            firstprob[h1] = sum_vec(bprob[0][h1].as_slice());
        }

        let mut tprob = vec![vec![vec![0.0 as f32; 1]; 1]; 1];
        if trans_prob_flag > 0 {
            tprob = vec![vec![vec![0.0 as f32; p]; p]; m];
            for h1 in 0..p {
                tprob[0][0][h1] = firstprob[h1];
            }
        }

        // Sample first position
        let (ind1, ind2) =
            constrained_paired_sample(firstprob.as_slice(), firstprob.as_slice(), rng);
        phase_ind[0][0] = ind1 as u8;
        phase_ind[1][0] = ind2 as u8;

        let psub = if trans_prob_flag == 0 { 2 } else { p };

        // Initialize forward prob
        let mut fprob_next = vec![vec![0.0 as f32; n]; psub];
        let mut fprob = vec![vec![0.0 as f32; n]; psub];

        // Emission at first position
        for h1 in 0..psub {
            for h2 in 0..n {
                let graph_ind = if trans_prob_flag == 0 {
                    phase_ind[h1][0] as usize
                } else {
                    h1
                };

                fprob[h1][h2] = emission_prob(x[0][h2], graph[0][graph_ind], eprob);
            }
        }

        for i in 1..m {
            // Combine fprob (from i-1) and bprob[i] to get transition probs
            let mut weights = vec![vec![0.0; p]; psub];

            for j in 0..psub {
                // Transition i-1 -> i
                let fsum = sum_vec(fprob[j].as_slice());
                for h2 in 0..n {
                    fprob_next[j][h2] =
                        transition_prob(fprob[j][h2], fsum, 1.0 / (n as f32), rprob);
                }

                // Multiply with backward prob and integrate to get transition prob from hap j to h1
                for h1 in 0..p {
                    let iprod = inner_prod(fprob_next[j].as_slice(), bprob[i][h1].as_slice());

                    // If not between blocks, set to identity matrix
                    let ind = if trans_prob_flag > 0 {
                        j
                    } else {
                        phase_ind[j][i - 1] as usize
                    };
                    if ind == h1 {
                        weights[j][h1] = mutex2f32(block_head[i], iprod, 1.0);
                    } else {
                        weights[j][h1] = mutex2f32(block_head[i], iprod, 0.0);
                    }
                }
            }

            let (h1, h2) = if trans_prob_flag > 0 {
                (phase_ind[0][i - 1] as usize, phase_ind[1][i - 1] as usize)
            } else {
                (0, 1)
            };

            let (ind1, ind2) =
                constrained_paired_sample(weights[h1].as_slice(), weights[h2].as_slice(), rng);

            // If i is NOT block head, then just keep the previous indices
            phase_ind[0][i] = mutex2u8(block_head[i], ind1 as u8, phase_ind[0][i - 1]);
            phase_ind[1][i] = mutex2u8(block_head[i], ind2 as u8, phase_ind[1][i - 1]);

            // Copy and renormalize
            if trans_prob_flag > 0 {
                let mut tsum = 0.0;
                if trans_prob_flag == 1 {
                    for h1 in 0..psub {
                        tsum = sum_vec(weights[h1].as_slice());
                        for h2 in 0..p {
                            tprob[i][h1][h2] = weights[h1][h2] / tsum;
                        }
                    }
                } else {
                    for h1 in 0..psub {
                        tsum += sum_vec(weights[h1].as_slice());
                    }

                    for h1 in 0..psub {
                        for h2 in 0..p {
                            tprob[i][h1][h2] = weights[h1][h2] / tsum;
                        }
                    }
                }
            }

            // Add emission at i (with the sampled haplotypes) to fprob_next
            for h1 in 0..psub {
                for h2 in 0..n {
                    let graph_ind = if trans_prob_flag == 0 {
                        phase_ind[h1][i] as usize
                    } else {
                        h1
                    };

                    fprob_next[h1][h2] *= emission_prob(x[i][h2], graph[i][graph_ind], eprob);
                }
            }

            // Update fprob
            for h1 in 0..psub {
                for h2 in 0..n {
                    fprob[h1][h2] = fprob_next[h1][h2];
                }
            }

            // Renormalize (TODO: replace with lazy normalization)
            let mut fsum = 0.0;
            for h1 in 0..psub {
                fsum += sum_vec(fprob[h1].as_slice());
            }
            for h1 in 0..psub {
                for h2 in 0..n {
                    fprob[h1][h2] /= fsum;
                }
            }
        }

        (phase_ind, tprob)
    }

    fn emission_prob(x_geno: u8, t_geno: u8, error_prob: f32) -> f32 {
        mutex2f32(x_geno == t_geno, 1.0 - error_prob, error_prob)
    }

    fn transition_prob(
        prev_prob: f32,
        total_prob: f32,
        uniform_frac: f32,
        recomb_prob: f32,
    ) -> f32 {
        prev_prob * (1.0 - recomb_prob) + total_prob * recomb_prob * uniform_frac
    }

    fn inner_prod(v1: &[f32], v2: &[f32]) -> f32 {
        let mut out = 0.0;
        for i in 0..v1.len() {
            out += v1[i] * v2[i];
        }
        out
    }

    fn sum_vec(v: &[f32]) -> f32 {
        let mut out = 0.0;
        for i in 0..v.len() {
            out += v[i]
        }
        out
    }

    // TODO make this side channel resilient
    fn sample(weights: &[f32], rng: &mut impl Rng) -> usize {
        let dist = WeightedIndex::new(&*weights).unwrap();
        dist.sample(rng)
    }

    // Only pairs of indices that add up to n (length of the weight vectors) are allowed
    // Weight of a pair is the product of the two weights (joint probability)
    fn constrained_paired_sample(
        weights1: &[f32],
        weights2: &[f32],
        rng: &mut impl Rng,
    ) -> (usize, usize) {
        let n = weights1.len();
        let mut combined = vec![0.0; n];
        for i in 0..n {
            combined[i] = weights1[i] * weights2[n - 1 - i];
        }
        let ind1 = sample(combined.as_slice(), rng);
        (ind1, n - 1 - ind1)
    }

    fn mutex2f32(b: bool, v1: f32, v0: f32) -> f32 {
        if b {
            return v1;
        } else {
            return v0;
        }
    }

    fn mutex2bool(b: bool, v1: bool, v0: bool) -> bool {
        if b {
            return v1;
        } else {
            return v0;
        }
    }

    fn mutex2(b: bool, v1: usize, v0: usize) -> usize {
        if b {
            return v1;
        } else {
            return v0;
        }
    }

    fn mutex2u8(b: bool, v1: u8, v0: u8) -> u8 {
        if b {
            return v1;
        } else {
            return v0;
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn genotype_graph_test() {
        let n = 20;
        let m = 200;

        // hmm parameters
        let rprob = 0.05; // recombination
        let eprob = 0.01; // error
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1234);

        let ref_x = (0..m)
            .map(|_| (0..n).map(|_| rng.gen_range(0..2)).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let t = (0..m)
            .map(|_| rng.gen_range(0..2) as u8)
            .collect::<Vec<_>>();

        let p = 1 << HET_PER_BLOCK;

        let (mut ref_graph, mut ref_head) =
            ref_algs::construct_geno_graph(t.as_slice(), HET_PER_BLOCK as i32);

        // Building

        #[cfg(feature = "leak-resist")]
        let (x, t) = (
            Array2::from_shape_fn((m, n), |(i, j)| Genotype::protect(ref_x[i][j] as i8)),
            Array1::from_shape_fn(m, |i| Genotype::protect(t[i] as i8)),
        );

        #[cfg(not(feature = "leak-resist"))]
        let (x, t) = (
            Array2::from_shape_fn((m, n), |(i, j)| ref_x[i][j] as i8),
            Array1::from_shape_fn(m, |i| t[i] as i8),
        );

        let mut graph = GenotypeGraph::build(t.view());

        #[cfg(feature = "leak-resist")]
        {
            let graph_block_head = graph
                .block_head
                .iter()
                .map(|v| v.expose())
                .collect::<Vec<_>>();
            let graph_graph = (0..m)
                .map(|i| {
                    (0..p)
                        .map(|j| graph.graph[[i, j]].expose() as u8)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            assert_eq!(graph_block_head, ref_head);
            assert_eq!(graph_graph, ref_graph);
        }

        #[cfg(not(feature = "leak-resist"))]
        {
            let graph_graph = (0..m)
                .map(|i| {
                    (0..p)
                        .map(|j| graph.graph[[i, j]] as u8)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            assert_eq!(graph.block_head.as_slice().unwrap(), ref_head.as_slice());
            assert_eq!(graph_graph, ref_graph);
        }

        // Pruning

        let ref_tprobs = (0..m)
            .map(|_| {
                (0..p)
                    .map(|_| (0..p).map(|_| rng.gen_range(0.0..1.1)).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        #[cfg(feature = "leak-resist")]
        let tprobs = Array3::from_shape_fn((m, p, p), |(i, j, k)| {
            Real::leaky_from_f32(ref_tprobs[i][j][k])
        });

        #[cfg(not(feature = "leak-resist"))]
        let tprobs = Array3::from_shape_fn((m, p, p), |(i, j, k)| ref_tprobs[i][j][k]);

        graph.prune(tprobs.view());

        ref_algs::prune_graph(&mut ref_graph, &mut ref_head, &ref_tprobs);

        #[cfg(feature = "leak-resist")]
        {
            let graph_block_head = graph
                .block_head
                .iter()
                .map(|v| v.expose())
                .collect::<Vec<_>>();
            let graph_graph = (0..m)
                .map(|i| {
                    (0..p)
                        .map(|j| graph.graph[[i, j]].expose() as u8)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            assert_eq!(graph_block_head, ref_head);
            assert_eq!(graph_graph, ref_graph);
        }

        #[cfg(not(feature = "leak-resist"))]
        {
            let graph_graph = (0..m)
                .map(|i| {
                    (0..p)
                        .map(|j| graph.graph[[i, j]] as u8)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            assert_eq!(graph.block_head.as_slice().unwrap(), ref_head.as_slice());
            assert_eq!(graph_graph, ref_graph);
        }

        // Backward pass
        let ref_bprob = ref_algs::backward_pass(&ref_x, &ref_graph, &ref_head, rprob, eprob);
        let bprob = graph.backward_pass(x.view(), rprob, eprob);

        for ((i, j), k) in (0..m).zip(0..p).zip(0..n) {
            #[cfg(feature = "leak-resist")]
            assert!((ref_bprob[i][j][k] - bprob[[i, j, k]].leaky_into_f32()).abs() < 1e-3);
            #[cfg(not(feature = "leak-resist"))]
            assert!((ref_bprob[i][j][k] - bprob[[i, j, k]]).abs() < f32::EPSILON);
        }

        // Forward sampling 0

        let mut seeded_rng = rand_chacha::ChaCha8Rng::seed_from_u64(1234);
        let mut ref_seeded_rng = rand_chacha::ChaCha8Rng::seed_from_u64(1234);

        let (phase_ind_1, tprobs_1) =
            graph.forward_sampling(x.view(), rprob, eprob, &mut seeded_rng, 0);
        let (ref_phase_ind_1, _) = ref_algs::forward_sampling(
            &ref_bprob,
            &ref_x,
            &ref_graph,
            &ref_head,
            rprob,
            eprob,
            &mut ref_seeded_rng,
            0,
        );

        assert!(tprobs_1.is_none());

        for (i, j) in (0..2).zip(0..m) {
            #[cfg(feature = "leak-resist")]
            assert_eq!(phase_ind_1[[i, j]].expose(), ref_phase_ind_1[i][j]);

            #[cfg(not(feature = "leak-resist"))]
            assert_eq!(phase_ind_1[[i, j]], ref_phase_ind_1[i][j]);
        }

        // Forward sampling 1

        let (phase_ind_2, tprobs_2) =
            graph.forward_sampling(x.view(), rprob, eprob, &mut seeded_rng, 1);
        let tprobs_2 = tprobs_2.unwrap();
        let (ref_phase_ind_2, ref_tprobs_2) = ref_algs::forward_sampling(
            &ref_bprob,
            &ref_x,
            &ref_graph,
            &ref_head,
            rprob,
            eprob,
            &mut ref_seeded_rng,
            1,
        );

        for (i, j) in (0..2).zip(0..m) {
            #[cfg(feature = "leak-resist")]
            assert_eq!(phase_ind_2[[i, j]].expose(), ref_phase_ind_2[i][j]);

            #[cfg(not(feature = "leak-resist"))]
            assert_eq!(phase_ind_2[[i, j]], ref_phase_ind_2[i][j]);
        }

        for ((i, j), k) in (0..m).zip(0..p).zip(0..p) {
            #[cfg(feature = "leak-resist")]
            assert!((tprobs_2[[i, j, k]].leaky_into_f32() - ref_tprobs_2[i][j][k]).abs() < 1e-2);

            #[cfg(not(feature = "leak-resist"))]
            assert!((tprobs_2[[i, j, k]] - ref_tprobs_2[i][j][k]).abs() < 1e-6);
        }

        // Forward sampling 2
        let (phase_ind_3, tprobs_3) =
            graph.forward_sampling(x.view(), rprob, eprob, &mut seeded_rng, 2);
        let tprobs_3 = tprobs_3.unwrap();
        let (ref_phase_ind_3, ref_tprobs_3) = ref_algs::forward_sampling(
            &ref_bprob,
            &ref_x,
            &ref_graph,
            &ref_head,
            rprob,
            eprob,
            &mut ref_seeded_rng,
            2,
        );

        for (i, j) in (0..2).zip(0..m) {
            #[cfg(feature = "leak-resist")]
            assert_eq!(phase_ind_3[[i, j]].expose(), ref_phase_ind_3[i][j]);

            #[cfg(not(feature = "leak-resist"))]
            assert_eq!(phase_ind_3[[i, j]], ref_phase_ind_3[i][j]);
        }

        for ((i, j), k) in (0..m).zip(0..p).zip(0..p) {
            #[cfg(feature = "leak-resist")]
            assert!((tprobs_3[[i, j, k]].leaky_into_f32() - ref_tprobs_3[i][j][k]).abs() < 1e-4);

            #[cfg(not(feature = "leak-resist"))]
            assert!((tprobs_3[[i, j, k]] - ref_tprobs_3[i][j][k]).abs() < 1e-6);
        }
    }

    #[test]
    fn top_k_test() {
        let mut rng = rand::thread_rng();
        let p = 1 << HET_PER_BLOCK;
        let ref_prob_matrix = (0..p)
            .map(|_| (0..p).map(|_| rng.gen_range(0.0..1.0)).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        #[cfg(feature = "leak-resist")]
        let prob_matrix =
            Array2::from_shape_fn((p, p), |(i, j)| Real::leaky_from_f32(ref_prob_matrix[i][j]));

        #[cfg(not(feature = "leak-resist"))]
        let prob_matrix = Array2::from_shape_fn((p, p), |(i, j)| ref_prob_matrix[i][j]);

        let (ref_ind1, ref_ind2, ref_sum) = ref_algs::select_top_k(&ref_prob_matrix, p);
        let (ind1, ind2, sum) = select_top_k(prob_matrix.view(), p);

        #[cfg(feature = "leak-resist")]
        let (ind1, ind2, sum) = (
            Array1::from_shape_fn(p, |i| ind1[i].expose()),
            Array1::from_shape_fn(p, |i| ind2[i].expose()),
            sum.leaky_into_f32(),
        );

        assert_eq!(ind1.as_slice().unwrap(), ref_ind1.as_slice());
        assert_eq!(ind2.as_slice().unwrap(), ref_ind2.as_slice());
        assert!((sum - ref_sum).abs() < 1e-3);
    }
}
