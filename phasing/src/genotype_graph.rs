use crate::{tp_value, Bool, Genotype, Real, UInt, U8};
use ndarray::{
    s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut1, ArrayViewMut2, 
    Zip,
};
use rand::Rng;
use std::time::Instant;

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum TransProbOption {
    Burnin,
    Prune,
    Main,
}

const P: usize = 8;

#[cfg(feature = "leak-resist")]
mod inner {
    use super::*;
    pub use crate::oram::SmallLSOram;
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

//#[cfg(feature = "leak-resist")]
//pub struct GenotypeGraph {
    //pub graph: Vec<SmallLSOram<Genotype>>,
    //pub block_head: Array1<Bool>,
//}

//#[cfg(not(feature = "leak-resist"))]
pub struct GenotypeGraph {
    pub graph: Array2<Genotype>,
    pub block_head: Array1<Bool>,
}

pub struct GenotypeGraphViewMut<'a> {
    pub graph: ArrayViewMut2<'a, Genotype>,
    pub block_head: ArrayViewMut1<'a, Bool>,
}

impl<'a> GenotypeGraphViewMut<'a> {
    pub fn forward_pass(
        &self,
        x: ArrayView2<Genotype>,
        rprobs: ArrayView1<Real>,
        rev_rprobs: ArrayView1<Real>,
        eprob: Real,
        rev_eprob: Real,
    ) -> Array3<Real> {
        let m = x.nrows();
        let n = x.ncols();

        let bprob = self.backward_pass(x, rprobs, rev_rprobs, eprob, rev_eprob);

        // p x p matrix at each pos i for transition between i-1 and i
        // Save belief over the first block in tprob[0][0]
        let firstprob =
            Array1::<Real>::from_shape_fn(P, |i| bprob.slice(s![0, i, ..]).iter().sum());

        let mut tprob = unsafe { Array3::<Real>::uninit((m, P, P)).assume_init() };
        for i in 0..P {
            tprob.slice_mut(s![0, i, ..]).assign(&firstprob);
        }

        // Initialize forward prob
        let mut fprob = unsafe { Array2::<Real>::uninit((P, n)).assume_init() };
        let mut fprob_next = unsafe { Array2::<Real>::uninit((P, n)).assume_init() };

        // Emission at first position
        for h1 in 0..P {
            for h2 in 0..n {
                fprob[[h1, h2]] =
                    emission_prob(x[[0, h2]], self.graph[[0, h1 as usize]], eprob, rev_eprob);
            }
        }

        let uniform_frac = 1.0 / (n as f32);

        #[cfg(feature = "leak-resist")]
        let uniform_frac = Real::protect_f32(uniform_frac);

        for i in 1..m {
            for j in 0..P {
                // Transition i-1 -> i
                let fsum = fprob.row(j).iter().sum();
                for h2 in 0..n {
                    fprob_next[[j, h2]] = transition_prob(
                        fprob[[j, h2]],
                        fsum,
                        uniform_frac.into(),
                        rprobs[i],
                        rev_rprobs[i],
                    );
                }
            }

            // block transition
            {
                // Add to aggregate sum over j
                let sums = Array1::from_shape_fn(n, |j| fprob_next.column(j).iter().sum());
                // If i-1 is block head, then replace fprob with its sum over j
                let prev_block_head = self.block_head[i - 1];
                Zip::from(fprob_next.rows_mut()).for_each(|mut a| {
                    Zip::from(&mut a).and(&sums).for_each(|b, c| {
                        #[cfg(feature = "leak-resist")]
                        {
                            *b = prev_block_head.select(*c, *b);
                        }
                        #[cfg(not(feature = "leak-resist"))]
                        {
                            if prev_block_head {
                                *b = *c;
                            }
                        }
                    });
                });
            }

            // Combine fprob (from i-1) and bprob[i] to get transition probs
            let mut weights = unsafe { Array2::<Real>::uninit((P, P)).assume_init() };
            for j in 0..P {
                // Multiply with backward prob and integrate to get transition prob from hap j to h1
                for h1 in 0..P {
                    // If not between blocks, set to identity matrix
                    let ind = tp_value!(j, u8);

                    #[cfg(feature = "leak-resist")]
                    {
                        let iprod = inner_prod(fprob_next.row(j), bprob.slice(s![i, h1, ..]));
                        #[cfg(feature = "leak-resist-fast")]
                        {
                            weights[[j, h1]] = ind.tp_eq(&U8::protect(h1 as u8)).select(
                                self.block_head[i].select(iprod, Real::protect_f32(1.0)),
                                self.block_head[i].select(iprod, Real::protect_f32(0.0)),
                            );
                        }
                        #[cfg(not(feature = "leak-resist-fast"))]
                        {
                            weights[[j, h1]] = ind.tp_eq(&U8::protect(h1 as u8)).select(
                                self.block_head[i].select(iprod, Real::protect_f32(1.0)),
                                self.block_head[i].select(iprod, Real::NAN),
                            );
                        }
                    }

                    #[cfg(not(feature = "leak-resist"))]
                    {
                        weights[[j, h1]] = if self.block_head[i] {
                            inner_prod(fprob_next.row(ind.into()), bprob.slice(s![i, h1, ..]))
                        } else {
                            if j as usize == h1 {
                                1.0
                            } else {
                                0.0
                            }
                        };
                    }
                }
            }

            tprob.slice_mut(s![i, .., ..]).assign(&weights);

            // Add emission at i (with the sampled haplotypes) to fprob_next
            for h1 in 0..P {
                for h2 in 0..n {
                    fprob_next[[h1, h2]] *=
                        emission_prob(x[[i, h2]], self.graph[[i, h1 as usize]], eprob, rev_eprob);
                }
            }

            // Renormalize
            let fsum = fprob_next.iter().sum::<Real>();
            fprob_next /= fsum;

            // Update fprob
            fprob.assign(&fprob_next);
        }

        tprob
    }

    fn backward_pass(
        &self,
        x: ArrayView2<Genotype>,
        rprobs: ArrayView1<Real>,
        rev_rprobs: ArrayView1<Real>,
        eprob: Real,
        rev_eprob: Real,
    ) -> Array3<Real> {
        let m = x.nrows();
        let n = x.ncols();

        let mut bprob = unsafe { Array3::<Real>::uninit((m, P, n)).assume_init() };

        // initialization (uniform over ref haplotypes + emission at last position)
        {
            #[cfg(feature = "leak-resist")]
            let row = Array1::from_vec(self.graph[m - 1].as_vec());

            #[cfg(not(feature = "leak-resist"))]
            let row = self.graph.row(m - 1);

            Zip::from(bprob.slice_mut(s![m - 1, .., ..]).rows_mut())
                .and(&row)
                .for_each(|mut a, &b| {
                    Zip::from(&mut a).and(x.row(m - 1)).for_each(|c, &d| {
                        *c = emission_prob(d, b, eprob, rev_eprob);
                    });
                });
            // renormalize
            let bsum: Real = bprob.slice(s![m - 1, .., ..]).iter().sum();
            bprob
                .slice_mut(s![m - 1, .., ..])
                .iter_mut()
                .for_each(|x| *x /= bsum);
        }

        let uniform_frac = 1.0 / (n as f32);

        #[cfg(feature = "leak-resist")]
        let uniform_frac = Real::protect_f32(uniform_frac);

        // backward pass
        for i in (0..m - 1).rev() {
            // i+1 -> i transition
            let (mut bprob_cur, bprob_next) =
                bprob.multi_slice_mut((s![i, .., ..], s![i + 1, .., ..]));
            Zip::from(bprob_cur.rows_mut())
                .and(bprob_next.rows())
                .for_each(|mut a, b| {
                    let sum: Real = b.iter().sum();
                    Zip::from(&mut a).and(&b).for_each(|c, &d| {
                        *c = transition_prob(d, sum, uniform_frac.into(), rprobs[i], rev_rprobs[i]);
                    });
                });

            // block transition
            {
                // Add to aggregate sum over h1
                let bprob_cur_t = bprob.slice(s![i, .., ..]);
                let sums = Array1::from_shape_fn(n, |j| bprob_cur_t.column(j).iter().sum());

                // If i+1 is block head, then replace bprob with its sum over h1
                let next_block_head = self.block_head[i + 1];
                Zip::from(bprob.slice_mut(s![i, .., ..]).rows_mut()).for_each(|mut a| {
                    Zip::from(&mut a).and(&sums).for_each(|b, c| {
                        #[cfg(feature = "leak-resist")]
                        {
                            *b = next_block_head.select(*c, *b);
                        }
                        #[cfg(not(feature = "leak-resist"))]
                        {
                            if next_block_head {
                                *b = *c;
                            }
                        }
                    });
                });
            }

            #[cfg(feature = "leak-resist")]
            let row = Array1::from_vec(self.graph[i].as_vec());

            #[cfg(not(feature = "leak-resist"))]
            let row = self.graph.row(i);
            // emission at i
            Zip::from(bprob.slice_mut(s![i, .., ..]).rows_mut())
                .and(&row)
                .for_each(|mut a, &b| {
                    Zip::from(&mut a).and(x.row(i)).for_each(|c, &d| {
                        *c *= emission_prob(d, b, eprob, rev_eprob);
                    });
                });

            // renormalize
            let bsum: Real = bprob.slice(s![i, .., ..]).iter().sum();
            bprob
                .slice_mut(s![i, .., ..])
                .iter_mut()
                .for_each(|x| *x /= bsum);
        }
        bprob
    }
}

pub const HET_PER_BLOCK: usize = 3;

impl GenotypeGraph {
    pub fn build(t: ArrayView1<Genotype>) -> Self {
        let m = t.len();
        let p = 1 << HET_PER_BLOCK;

        #[cfg(feature = "leak-resist")]
        let mut graph = Vec::with_capacity(m);

        #[cfg(not(feature = "leak-resist"))]
        let mut graph = unsafe { Array2::<Genotype>::uninit((m, p)).assume_init() };

        let mut block_head = unsafe { Array1::<Bool>::uninit(m).assume_init() };

        let mut cur_het_count = tp_value!(HET_PER_BLOCK - 1, u32);

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
                let mut col = Vec::with_capacity(p);
                for j in 0..p {
                    col.push(cur_geno);
                    let (_, modulo) = oram_sgx::utils::tp_u32_div(
                        UInt::protect(j as u32 + 1),
                        step_size,
                        p as u32,
                    );
                    cur_geno = modulo.tp_eq(&0).select(cur_geno + 1 & 1, cur_geno);
                }
                graph.push(SmallLSOram::from_slice(&col));
            }

            #[cfg(not(feature = "leak-resist"))]
            {
                if t[i] == 1 {
                    cur_het_count += 1;
                    if cur_het_count == HET_PER_BLOCK as u32 {
                        block_head[i] = true;
                        cur_het_count = 0;
                    } else {
                        block_head[i] = false;
                    }
                    for j in 0..p {
                        graph[[i, j]] = ((j >> cur_het_count) & 1) as i8;
                    }
                } else {
                    block_head[i] = false;
                    for j in 0..p {
                        graph[[i, j]] = t[i] / 2;
                    }
                }

                //let head_flag = cur_het_count == HET_PER_BLOCK as u32;
                //block_head[i] = head_flag;
                //// Reset if at block head
                //if head_flag {
                //cur_het_count = 0;
                //}
                //let het_flag = t[i] == 1;
                //if het_flag {
                //cur_het_count += 1;
                //}

                //let start_geno = if t[i] < 2 { 0 } else { 1 };

                //let step_size = if het_flag {
                //1 << (cur_het_count - 1)
                //} else {
                //p
                //};
                //let mut cur_geno = start_geno;
                //for j in 0..p {
                //graph[[i, j]] = cur_geno;
                //if ((j + 1) % step_size) == 0 {
                //cur_geno = (cur_geno + 1) % 2;
                //}
                //}
            }
        }

        block_head[0] = tp_value!(false, bool);

        Self {
            graph,
            block_head,
        }
    }

    pub fn subview_mut<'a>(&'a mut self, start: usize, end: usize) -> GenotypeGraphViewMut<'a> {
        GenotypeGraphViewMut {
            graph: self.graph.slice_mut(s![start..end, ..]),
            block_head: self.block_head.slice_mut(s![start..end]),
        }
    }

    // TODO: limit merges to maximum of MAX_AMBIGUOUS het sites within a block
    pub fn prune(&mut self, tprob: ArrayView3<Real>) {
        let now = Instant::now();
        const MCMC_PRUNE_PROB_THRES: f32 = 0.999; // "mcmc-prune" parameter in ShapeIt4
        const MAX_AMBIGUOUS: usize = 22; // "MAX_AMB" constant in ShapeIt4 (utils/otools.h)

        #[cfg(feature = "leak-resist")]
        let m = tprob.shape()[0];

        #[cfg(not(feature = "leak-resist"))]
        let m = tprob.shape()[0];

        // Backward pass to identify where to merge adjacent blocks and the new haps
        let mut ind1_cache = unsafe { Array2::<U8>::uninit((m, P)).assume_init() };
        let mut ind2_cache = unsafe { Array2::<U8>::uninit((m, P)).assume_init() };
        let mut merge_head = vec![tp_value!(false, bool); m];
        let mut merge_flag = tp_value!(false, bool);
        let mut new_merge_flag;

        for i in (0..m - 1).rev() {
            let (ind1, ind2, prob) = select_top_k(tprob.slice(s![i, .., ..]), P);
            // If merge flag set then carry over
            for j in 0..P {
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
                    & prob.tp_gt(&Real::protect_f32(MCMC_PRUNE_PROB_THRES))
                    & !merge_flag;

                // If at head then merge_flag is set to new_merge_flag, otherwise carry over
                merge_flag = self.block_head[i].select(new_merge_flag, merge_flag);
            }

            #[cfg(not(feature = "leak-resist"))]
            {
                // If at head and merge is on, it is the merge head
                merge_head[i] = (i == 0 || self.block_head[i]) && merge_flag;

                // If merge is not on, we're at a head and prob over threshold, start a new merge
                new_merge_flag =
                    self.block_head[i] && prob > MCMC_PRUNE_PROB_THRES as Real && !merge_flag;

                // If at head then merge_flag is set to new_merge_flag, otherwise carry over
                if self.block_head[i] {
                    merge_flag = new_merge_flag;
                }
            }
        }

        // Forward pass to update the graph and block_head
        let mut new_geno = unsafe { Array1::<Genotype>::uninit(P).assume_init() };
        let mut ind1 = unsafe { Array1::<U8>::uninit(P).assume_init() };
        let mut ind2 = unsafe { Array1::<U8>::uninit(P).assume_init() };
        let mut ind = unsafe { Array1::<U8>::uninit(P).assume_init() };
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

                for j in 0..P {
                    new_geno[j] = self.graph[i].obliv_read(ind[j].as_u32());
                }
                self.graph[i] = SmallLSOram::from_slice(new_geno.as_slice().unwrap());

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

                for j in 0..P {
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

                for j in 0..P {
                    new_geno[j] = self.graph[[i, ind[j] as usize]];
                }
                for j in 0..P {
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

    pub fn get_haps(&self, phase_ind: ArrayView2<U8>) -> Array2<Genotype> {

        //#[cfg(feature = "leak-resist")]
        //let m = phase_ind.nrows();

        //#[cfg(not(feature = "leak-resist"))]
        let m = phase_ind.nrows();

        let mut phased = unsafe { Array2::<Genotype>::uninit((m, 2)).assume_init() };

        for i in 0..m {
            for hap in 0..2 {
                //#[cfg(feature = "leak-resist")]
                //{
                    //phased[[hap, i]] = self.graph[i].obliv_read(phase_ind[[hap, i]].as_u32());
                //}

                //#[cfg(not(feature = "leak-resist"))]
                {
                    phased[[i, hap]] = self.graph[[i, phase_ind[[i, hap]] as usize]];
                }
            }
        }
        phased
    }
}

// Takes a p x p matrix M and returns a list of 0-based
// index pairs (i,j) corresponding to top K largest
// elements according to M(i,j) * M(p-1-i,p-1-j)
// To avoid duplicates, restrict i <= p/2
// Also output associated probability mass
fn select_top_k(tab: ArrayView2<Real>, k: usize) -> (Array1<U8>, Array1<U8>, Real) {
    // Normalize
    let tab = &tab / tab.sum();

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
    rng: impl Rng,
) -> (UInt, UInt) {
    let n = weights1.len();
    //let weights1 = {
    //let weights_sum = weights1.sum();
    //&weights1 / weights_sum
    //};

    //let weights2 = {
    //let weights_sum = weights2.sum();
    //&weights2 / weights_sum
    //};

    let mut combined = unsafe { Array1::<Real>::uninit(n).assume_init() };
    for i in 0..n {
        #[cfg(feature = "leak-resist")]
        {
            #[cfg(not(feature = "leak-resist-fast"))]
            {
                combined[i] = (weights1[i].is_nan() | weights2[n - 1 - i].is_nan())
                    .select(Real::NAN, weights1[i] * weights2[n - 1 - i]);
            }
            #[cfg(feature = "leak-resist-fast")]
            {
                combined[i] = weights1[i] * weights2[n - 1 - i];
            }
        }

        #[cfg(not(feature = "leak-resist"))]
        {
            combined[i] = weights1[i] * weights2[n - 1 - i];
        }
    }
    let ind1 = weighted_sample(combined.view(), rng);
    (ind1, tp_value!(n, u32) - 1 - ind1)
}

fn weighted_sample(weights: ArrayView1<Real>, mut rng: impl Rng) -> UInt {
    let mut total_weight = weights[0];
    let mut cumulative_weights: Vec<Real> = Vec::with_capacity(weights.len());
    cumulative_weights.push(total_weight);
    for &w in weights.iter().skip(1) {
        #[cfg(feature = "leak-resist")]
        {
            #[cfg(not(feature = "leak-resist-fast"))]
            {
                total_weight = w.is_nan().select(
                    total_weight,
                    total_weight.is_nan().select(w, total_weight + w),
                );
            }
            #[cfg(feature = "leak-resist-fast")]
            {
                total_weight += w;
            }
        }
        #[cfg(not(feature = "leak-resist"))]
        {
            total_weight += w;
        }
        cumulative_weights.push(total_weight);
    }

    #[cfg(feature = "leak-resist")]
    {
        let chosen_weight = Real::protect_f32(rng.gen_range(0.0..1.0)) * total_weight;
        let mut index = UInt::protect(0);
        let mut done = Bool::protect(false);
        for w in cumulative_weights {
            #[cfg(not(feature = "leak-resist-fast"))]
            {
                done = (w.tp_gt(&chosen_weight) & !w.is_nan()).select(Bool::protect(true), done);
            }
            #[cfg(feature = "leak-resist-fast")]
            {
                done = w.tp_gt(&chosen_weight).select(Bool::protect(true), done);
            }
            index = (!done).select(index + 1, index);
        }
        index
            .tp_eq(&(weights.len() as u32))
            .select(UInt::protect(0), index)
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

#[inline(always)]
fn emission_prob(
    x_geno: Genotype,
    t_geno: Genotype,
    error_prob: Real,
    rev_error_prob: Real,
) -> Real {
    #[cfg(feature = "leak-resist")]
    {
        x_geno.tp_eq(&t_geno).select(rev_error_prob, error_prob)
    }

    #[cfg(not(feature = "leak-resist"))]
    {
        if x_geno == t_geno {
            rev_error_prob
        } else {
            error_prob
        }
    }
}

#[inline(always)]
fn transition_prob(
    prev_prob: Real,
    total_prob: Real,
    uniform_frac: Real,
    recomb_prob: Real,
    rev_recomb_prob: Real,
) -> Real {
    prev_prob * rev_recomb_prob + total_prob * recomb_prob * uniform_frac
}

fn inner_prod(v1: ArrayView1<Real>, v2: ArrayView1<Real>) -> Real {
    let prod = &v1 * &v2;
    prod.into_iter().sum()
}

#[cfg(test)]
mod test {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn genotype_graph_node_test() {
        let n = 20;
        let m = 200;

        let ref_rprob = 0.05;
        let ref_eprob = 0.01;
        // hmm parameters
        let rprob = ref_rprob; // recombination
        let rev_rprob = 1. - ref_rprob; // recombination
        let eprob = ref_eprob; // error
        let rev_eprob = 1. - eprob; // error

        #[cfg(feature = "leak-resist")]
        let (rprob, rev_rprob, eprob, rev_eprob) = (
            Real::protect_f32(rprob),
            Real::protect_f32(rev_rprob),
            Real::protect_f32(eprob),
            Real::protect_f32(rev_eprob),
        );

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1234);

        let ref_x = (0..m)
            .map(|_| (0..n).map(|_| rng.gen_range(0..2)).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let t = (0..m)
            .map(|_| rng.gen_range(0..2) as u8)
            .collect::<Vec<_>>();

        let p = 1 << HET_PER_BLOCK;

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

        Zip::from(&graph.block_head)
            .and(graph.graph.rows())
            .for_each(|b, g| {
                let sum = g.sum();
                if *b {
                    assert_eq!(sum, 4);
                } else {
                    assert!((sum == 8 || sum == 0 || sum == 4));
                }
            });
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
            Array2::from_shape_fn((p, p), |(i, j)| Real::protect_f32(ref_prob_matrix[i][j]));

        #[cfg(not(feature = "leak-resist"))]
        let prob_matrix = Array2::<Real>::from_shape_fn((p, p), |(i, j)| ref_prob_matrix[i][j]);

        let (ref_ind1, ref_ind2, ref_sum) = ref_algs::select_top_k(&ref_prob_matrix, p);
        let (ind1, ind2, sum) = select_top_k(prob_matrix.view(), p);

        #[cfg(feature = "leak-resist")]
        let (ind1, ind2, sum) = (
            Array1::from_shape_fn(p, |i| ind1[i].expose()),
            Array1::from_shape_fn(p, |i| ind2[i].expose()),
            sum.expose_into_f32(),
        );

        assert_eq!(ind1.as_slice().unwrap(), ref_ind1.as_slice());
        assert_eq!(ind2.as_slice().unwrap(), ref_ind2.as_slice());
        assert!((sum - ref_sum).abs() < 1e-3);
    }

    ////#[test]
    //fn backward_pass_bench() {
    //let n = 1000;
    //let m = 10000;

    //// hmm parameters
    //let rprob = 0.05; // recombination
    //let eprob = 0.01; // error
    //let rev_rprob = 1. - rprob; // recombination
    //let rev_eprob = 1. - eprob; // error

    //#[cfg(feature = "leak-resist")]
    //let (rprob, rev_rprob, eprob, rev_eprob) = (
    //Real::protect_f32(rprob),
    //Real::protect_f32(rev_rprob),
    //Real::protect_f32(eprob),
    //Real::protect_f32(rev_eprob),
    //);
    //let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1234);

    //let ref_x = (0..m)
    //.map(|_| (0..n).map(|_| rng.gen_range(0..2)).collect::<Vec<_>>())
    //.collect::<Vec<_>>();

    //let t = (0..m)
    //.map(|_| rng.gen_range(0..2) as u8)
    //.collect::<Vec<_>>();

    //// Building

    //#[cfg(feature = "leak-resist")]
    //let (x, t) = (
    //Array2::from_shape_fn((m, n), |(i, j)| Genotype::protect(ref_x[i][j] as i8)),
    //Array1::from_shape_fn(m, |i| Genotype::protect(t[i] as i8)),
    //);

    //#[cfg(not(feature = "leak-resist"))]
    //let (x, t) = (
    //Array2::from_shape_fn((m, n), |(i, j)| ref_x[i][j] as i8),
    //Array1::from_shape_fn(m, |i| t[i] as i8),
    //);

    //let graph = GenotypeGraph::build(t.view());

    //graph.backward_pass(x.view(), rprob, rev_rprob, eprob, rev_eprob);
    //}

    ////#[test]
    //fn forward_sampling_bench() {
    //let n = 10000;
    //let m = 100;

    //// hmm parameters
    //let rprob = 0.05; // recombination
    //let eprob = 0.01; // error
    //let rev_rprob = 1. - rprob; // recombination
    //let rev_eprob = 1. - eprob; // error

    //#[cfg(feature = "leak-resist")]
    //let (rprob, rev_rprob, eprob, rev_eprob) = (
    //Real::protect_f32(rprob),
    //Real::protect_f32(rev_rprob),
    //Real::protect_f32(eprob),
    //Real::protect_f32(rev_eprob),
    //);
    //let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1234);

    //let ref_x = (0..m)
    //.map(|_| (0..n).map(|_| rng.gen_range(0..2)).collect::<Vec<_>>())
    //.collect::<Vec<_>>();

    //let t = (0..m)
    //.map(|_| rng.gen_range(0..2) as u8)
    //.collect::<Vec<_>>();

    //// Building

    //#[cfg(feature = "leak-resist")]
    //let (x, t) = (
    //Array2::from_shape_fn((m, n), |(i, j)| Genotype::protect(ref_x[i][j] as i8)),
    //Array1::from_shape_fn(m, |i| Genotype::protect(t[i] as i8)),
    //);

    //#[cfg(not(feature = "leak-resist"))]
    //let (x, t) = (
    //Array2::from_shape_fn((m, n), |(i, j)| ref_x[i][j] as i8),
    //Array1::from_shape_fn(m, |i| t[i] as i8),
    //);

    //let graph = GenotypeGraph::build(t.view());

    //graph.forward_sampling(
    //x.view(),
    //rprob,
    //rev_rprob,
    //eprob,
    //rev_eprob,
    //&mut rng,
    //TransProbOption::Burnin,
    //);
    //}
}
