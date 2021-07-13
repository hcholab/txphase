use crate::{Bool, Genotype, Real, UInt, U8};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ArrayView3};

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

#[cfg(not(feature = "leak-resist"))]
mod inner {}

use inner::*;

pub struct GenotypeGraph {
    pub graph: Array2<Genotype>,
    pub block_head: Array1<Bool>,
    pub p: usize,
}

impl GenotypeGraph {
    pub fn build(t: ArrayView1<Genotype>, het_per_block: usize) -> Self {
        let m = t.len();
        let p = 1 << het_per_block;
        let mut graph = unsafe { Array2::<Genotype>::uninit((m, p)).assume_init() };
        let mut block_head = unsafe { Array1::<Bool>::uninit(m).assume_init() };

        #[cfg(feature = "leak-resist")]
        {
            block_head[0] = Bool::protect(false); // Skip first position
        }

        #[cfg(not(feature = "leak-resist"))]
        {
            block_head[0] = false; // Skip first position
        }

        #[cfg(feature = "leak-resist")]
        let mut cur_het_count = UInt::protect(het_per_block as u32);

        #[cfg(not(feature = "leak-resist"))]
        let mut cur_het_count = het_per_block;

        for i in 0..m {
            #[cfg(feature = "leak-resist")]
            {
                let head_flag = cur_het_count.tp_eq(&(het_per_block as u32));
                block_head[i] = head_flag;
                cur_het_count = head_flag.select(cur_het_count + 1, cur_het_count);

                let het_flag = t[i].tp_eq(&1);
                let start_geno = t[i]
                    .tp_lt(&2)
                    .select(Genotype::protect(0), Genotype::protect(1));
                let step_size = het_flag.select(
                    oram_sgx::utils::tp_u32_shl(
                        UInt::protect(1),
                        cur_het_count - 1,
                        het_per_block as u32,
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
                let head_flag = cur_het_count == het_per_block;
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

                let step_size = if het_flag { 1 << cur_het_count - 1 } else { p };
                let mut cur_geno = start_geno;
                for j in 0..p {
                    graph[[i, j]] = cur_geno;
                    if ((j + 1) % step_size) == 0 {
                        cur_geno = (cur_geno + 1) % 2;
                    }
                }
            }
        }

        Self {
            graph,
            block_head,
            p,
        }
    }

    // TODO: limit merges to maximum of MAX_AMBIGUOUS het sites within a block
    pub fn prune(&mut self, tprob: ArrayView3<Real>) {
        const MCMC_PRUNE_PROB_THRES: f32 = 0.999; // "mcmc-prune" parameter in ShapeIt4
        const MAX_AMBIGUOUS: usize = 22; // "MAX_AMB" constant in ShapeIt4 (utils/otools.h)

        let m = self.graph.nrows();
        let p = self.p;

        // Backward pass to identify where to merge adjacent blocks and the new haps
        let mut ind1_cache = unsafe { Array2::<U8>::uninit((m, p)).assume_init() };
        let mut ind2_cache = unsafe { Array2::<U8>::uninit((m, p)).assume_init() };
        let mut merge_head;
        let mut merge_flag;
        let mut new_merge_flag;

        #[cfg(feature = "leak-resist")]
        {
            merge_head = vec![Bool::protect(false); m];
            merge_flag = Bool::protect(false);
        }

        #[cfg(not(feature = "leak-resist"))]
        {
            merge_head = vec![false; m];
            merge_flag = false;
        }

        for i in (0..m - 1).rev() {
            let (ind1, ind2, prob) = Self::select_top_k(tprob.slice(s![i, .., ..]), p);

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
        let mut block_counter;

        #[cfg(feature = "leak-resist")]
        {
            block_counter = UInt::protect(0);
        }

        #[cfg(not(feature = "leak-resist"))]
        {
            block_counter = 0;
        }

        for i in 0..m {
            #[cfg(feature = "leak-resist")]
            {
                block_counter = merge_head[i].select(UInt::protect(2), block_counter);
                block_counter = (!merge_head[i] & self.block_head[i]).select(
                    (block_counter)
                        .tp_gt(&1)
                        .select(block_counter - 1, UInt::protect(0)),
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
    }

    // TODO make oblivious
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
                    //elems.push((dip, U8::protect(i as u8), U8::protect(j as u8)));
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
}
