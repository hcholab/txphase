use crate::{tp_value, Genotype, Real, U8};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut2, Zip};
use std::time::Instant;

pub const HET_PER_SEGMENT: usize = 3;
pub const P: usize = 1 << HET_PER_SEGMENT;

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

#[derive(Copy, Clone)]
pub struct G(U8);

impl G {
    pub fn new_het(het_count: u32) -> Self {
        Self(match het_count {
            0 => 0b01010101,
            1 => 0b00110011,
            2 => 0b00001111,
            _ => panic!("Invalid het count"),
        })
    }

    pub fn new_hom(hom: Genotype) -> Self {
        Self(match hom {
            0 => 0b00000000,
            2 => 0b11111111,
            _ => panic!("Invalid homozygote"),
        })
    }

    #[inline]
    pub fn get_row(self, i: usize) -> Genotype {
        (self.0 >> i & 1) as Genotype
    }

    #[inline]
    pub fn set_row(&mut self, i: usize, genotype: i8) {
        if genotype == 1 {
            self.0 |= 1 << i;
        } else if genotype == 0 {
            self.0 &= !(1 << i);
        } else {
            panic!("Invalid genotype");
        }
    }

    #[inline]
    pub fn is_segment_marker(self) -> bool {
        self.0 == 0b01010101
    }
}

//#[cfg(not(feature = "leak-resist"))]
pub struct GenotypeGraph {
    pub graph: Array1<G>,
}

impl GenotypeGraph {
    pub fn build(t: ArrayView1<Genotype>) -> Self {
        let m = t.len();

        let mut graph = unsafe { Array1::<G>::uninit(m).assume_init() };

        let mut cur_het_count = tp_value!(HET_PER_SEGMENT - 1, u32);

        for i in 0..m {
            if t[i] == 1 {
                cur_het_count += 1;
                if cur_het_count == HET_PER_SEGMENT as u32 {
                    cur_het_count = 0;
                } else {
                }
                graph[i] = G::new_het(cur_het_count);
            } else {
                graph[i] = G::new_hom(t[i]);
            }
        }

        Self { graph }
    }

    pub fn slice<'a>(&'a mut self, start: usize, end: usize) -> GenotypeGraphSlice<'a> {
        GenotypeGraphSlice {
            graph: self.graph.slice(s![start..end]),
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
                let is_segment_marker = self.graph[i].is_segment_marker();
                // If at head and merge is on, it is the merge head
                merge_head[i] = (i == 0 || is_segment_marker) && merge_flag;

                // If merge is not on, we're at a head and prob over threshold, start a new merge
                new_merge_flag =
                    is_segment_marker && prob > MCMC_PRUNE_PROB_THRES as Real && !merge_flag;

                // If at head then merge_flag is set to new_merge_flag, otherwise carry over
                if is_segment_marker {
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
                let is_segment_marker = self.graph[i].is_segment_marker();
                // At merge head set block counter to 2 (will update the next two blocks)
                if merge_head[i] {
                    block_counter = 2;
                }
                // Every time we see a block head that is not a merge head, decrement block counter
                if !merge_head[i] && is_segment_marker {
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
                    new_geno[j] = self.graph[i].get_row(ind[j] as usize);
                }
                for j in 0..P {
                    self.graph[i].set_row(j, new_geno[j]);
                }

                //// Erase block_head flag between the two blocks being merged
                //if self.block_head[i] && block_counter == 1 {
                //self.block_head[i] = false;
                //}
            }
        }
        println!(
            "Genotype graph pruning: {} ms",
            (Instant::now() - now).as_millis()
        );
    }

    pub fn traverse_graph_pair(
        &self,
        ind: ArrayView2<u8>,
        mut haps: ArrayViewMut2<Genotype>,
    ) {
        Zip::from(haps.rows_mut())
            .and(ind.rows())
            .and(&self.graph)
            .for_each(|mut h_row, ind_row, g| {
                h_row[0] = g.get_row(ind_row[0] as usize);
                h_row[1] = g.get_row(ind_row[1] as usize);
            });
    }
}

pub struct GenotypeGraphSlice<'a> {
    pub graph: ArrayView1<'a, G>,
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
}
