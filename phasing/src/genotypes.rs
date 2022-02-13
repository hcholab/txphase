use crate::{tp_value, Genotype, Real, U8};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut1, ArrayViewMut2, Zip};

pub const HET_PER_SEGMENT: usize = 3;
pub const P: usize = 1 << HET_PER_SEGMENT;

#[repr(i8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum G {
    HomRef = 0,
    HomAlt = 2,
    Het1 = -1,
    Het2 = -2,
    Het3 = -3,
}

impl G {
    #[inline]
    pub fn access_graph_row(self, i: u8) -> i8 {
        let g = self as i8;
        (if g < 0 {
            (i >> (-g as u8)) & 1
        } else {
            g as u8 / 2
        }) as i8
    }

    #[inline]
    pub fn is_segment_marker(self) -> bool {
        self == Self::Het3
    }
}

impl From<i8> for G {
    fn from(g: i8) -> Self {
        match g {
            0 => Self::HomRef,
            2 => Self::HomAlt,
            -1 => Self::Het1,
            -2 => Self::Het2,
            -3 => Self::Het3,
            _ => panic!("Invalid genotype"),
        }
    }
}

impl Into<i8> for G {
    fn into(self) -> i8 {
        self as i8
    }
}

pub struct Genotypes;

impl Genotypes {
    pub fn build(genotypes: ArrayView1<Genotype>) -> Array1<G> {
        let mut het_counter = 0i8;
        let mut out = Vec::<G>::with_capacity(genotypes.len());
        for &g in genotypes {
            if g == 1 {
                het_counter += 1;
                out.push((-het_counter).into());
                if het_counter == HET_PER_SEGMENT as i8 {
                    het_counter = 0;
                }
            } else {
                out.push(g.into());
            }
        }
        Array1::from_vec(out)
    }

    pub fn traverse_graph_pair(
        genotypes: ArrayView1<G>,
        ind: ArrayView2<u8>,
        mut haps: ArrayViewMut2<Genotype>,
    ) {
        Zip::from(haps.rows_mut())
            .and(ind.rows())
            .and(&genotypes)
            .for_each(|mut h_row, ind_row, t| {
                h_row[0] = t.access_graph_row(ind_row[0]);
                h_row[1] = t.access_graph_row(ind_row[1]);
            });
    }

    // TODO: limit merges to maximum of MAX_AMBIGUOUS het sites within a block
    pub fn prune(tprobs: ArrayView3<Real>, genotypes: ArrayViewMut1<G>) {
        const MCMC_PRUNE_PROB_THRES: f32 = 0.999; // "mcmc-prune" parameter in ShapeIt4
        const MAX_AMBIGUOUS: usize = 22; // "MAX_AMB" constant in ShapeIt4 (utils/otools.h)

        #[cfg(feature = "leak-resist")]
        let m = tprobs.shape()[0];

        #[cfg(not(feature = "leak-resist"))]
        let m = tprobs.shape()[0];

        // Backward pass to identify where to merge adjacent blocks and the new haps
        let mut ind1_cache = unsafe { Array2::<U8>::uninit((m, P)).assume_init() };
        let mut ind2_cache = unsafe { Array2::<U8>::uninit((m, P)).assume_init() };
        let mut merge_head = vec![tp_value!(false, bool); m];
        let mut merge_flag = tp_value!(false, bool);
        let mut new_merge_flag;

        for i in (0..m - 1).rev() {
            let (ind1, ind2, prob) = select_top_k(tprobs.slice(s![i, .., ..]), P);
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

            //#[cfg(feature = "leak-resist")]
            //{
                //// If at head and merge is on, it is the merge head
                //merge_head[i] = (self.block_head[i] | (i == 0)) & merge_flag;

                //// If merge is not on, we're at a head and prob over threshold, start a new merge
                //new_merge_flag = self.block_head[i]
                    //& prob.tp_gt(&Real::protect_f32(MCMC_PRUNE_PROB_THRES))
                    //& !merge_flag;

                //// If at head then merge_flag is set to new_merge_flag, otherwise carry over
                //merge_flag = self.block_head[i].select(new_merge_flag, merge_flag);
            //}

            //#[cfg(not(feature = "leak-resist"))]
            //{
            // If at head and merge is on, it is the merge head
            merge_head[i] = (i == 0 || genotypes[i].is_segment_marker()) && merge_flag;

            // If merge is not on, we're at a head and prob over threshold, start a new merge
            new_merge_flag =
                genotypes[i].is_segment_marker() && prob > MCMC_PRUNE_PROB_THRES as Real && !merge_flag;

            // If at head then merge_flag is set to new_merge_flag, otherwise carry over
            if genotypes[i].is_segment_marker() {
                merge_flag = new_merge_flag;
            }
            //}
        }

        // Forward pass to update the graph and block_head
        let mut new_geno = unsafe { Array1::<Genotype>::uninit(P).assume_init() };
        let mut ind1 = unsafe { Array1::<U8>::uninit(P).assume_init() };
        let mut ind2 = unsafe { Array1::<U8>::uninit(P).assume_init() };
        let mut ind = unsafe { Array1::<U8>::uninit(P).assume_init() };
        let mut block_counter = tp_value!(0, i8);

        for i in 0..m {
            //#[cfg(feature = "leak-resist")]
            //{
                //block_counter = merge_head[i].select(tp_value!(2, i8), block_counter);
                //block_counter = (!merge_head[i] & self.block_head[i]).select(
                    //(block_counter)
                        //.tp_gt(&1)
                        //.select(block_counter - 1, tp_value!(0, i8)),
                    //block_counter,
                //);
                //for j in 0..p {
                    //// If at merge head copy cache over
                    //ind1[j] = merge_head[i].select(ind1_cache[[i, j]], ind1[j]);
                    //ind2[j] = merge_head[i].select(ind2_cache[[i, j]], ind2[j]);

                    //// If block_counter hits zero change back to original indicies
                    //ind1[j] = block_counter
                        //.tp_eq(&0)
                        //.select(U8::protect(j as u8), ind1[j]);
                    //ind2[j] = block_counter
                        //.tp_eq(&0)
                        //.select(U8::protect(j as u8), ind2[j]);

                    //// Use ind1 for first block, ind2 for second block
                    //ind[j] = block_counter.tp_eq(&2).select(ind1[j], ind2[j]);
                //}

                //for j in 0..P {
                    //new_geno[j] = self.graph[i].obliv_read(ind[j].as_u32());
                //}
                //self.graph[i] = SmallLSOram::from_slice(new_geno.as_slice().unwrap());

                //// Erase block_head flag between the two blocks being merged
                //self.block_head[i] = (self.block_head[i] & block_counter.tp_eq(&1))
                    //.select(Bool::protect(false), self.block_head[i]);
            //}

            //#[cfg(not(feature = "leak-resist"))]
            //{
            // At merge head set block counter to 2 (will update the next two blocks)
            if merge_head[i] {
                block_counter = 2;
            }
            // Every time we see a block head that is not a merge head, decrement block counter
            if !merge_head[i] && genotypes[i].is_segment_marker() {
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
                new_geno[j] = genotypes[i].access_graph_row(ind[j]);
            }

            todo!()
            //for j in 0..P {
                //self.graph[[i, j]] = new_geno[j];
            //}

            //// Erase block_head flag between the two blocks being merged
            //if self.block_head[i] && block_counter == 1 {
                //self.block_head[i] = false;
            //}
        }
        //}
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
