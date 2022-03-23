use crate::{tp_value, Genotype, Real, U8};
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut2, Zip};

pub const HET_PER_SEGMENT: usize = 3;
pub const P: usize = 1 << HET_PER_SEGMENT;
pub const MCMC_PRUNE_PROB_THRES: f64 = 0.999; // "mcmc-prune" parameter in ShapeIt4
pub const MAX_HETS: usize = 22; // "MAX_AMB" constant in ShapeIt4 (utils/otools.h)

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

#[derive(Clone, Copy)]
pub struct GMeta(U8);

const SEGMENT_MARKER_MASK: u8 = 0b11111110;

impl GMeta {
    fn new(is_segment_marker: bool, is_het: bool, is_alt_allele: bool) -> Self {
        let mut inner = 0u8;
        inner |= is_segment_marker as u8;
        inner |= (is_het as u8) << 1;
        inner |= (is_alt_allele as u8) << 2;
        Self(inner)
    }

    #[inline]
    fn get_segment_marker(self) -> bool {
        self.0 & 1 == 1
    }

    #[inline]
    fn get_het(self) -> bool {
        ((self.0 >> 1) & 1) == 1
    }

    #[inline]
    fn unset_segment_marker(&mut self) {
        self.0 &= SEGMENT_MARKER_MASK
    }
}

#[derive(Copy, Clone)]
pub struct G {
    graph: U8,
    meta: GMeta,
}

impl G {
    pub fn new_het(het_count: u32) -> Self {
        let (graph, segment_marker) = match het_count {
            0 => (0b01010101, true),
            1 => (0b00110011, false),
            2 => (0b00001111, false),
            _ => panic!("Invalid het count"),
        };
        Self {
            graph,
            meta: GMeta::new(segment_marker, true, false),
        }
    }

    pub fn new_hom(hom: Genotype) -> Self {
        let (graph, segment_marker) = match hom {
            0 => (0b00000000, false),
            2 => (0b11111111, false),
            _ => panic!("Invalid homozygote"),
        };
        Self {
            graph,
            meta: GMeta::new(segment_marker, false, hom == 2),
        }
    }

    #[inline]
    pub fn get_row(&self, i: usize) -> Genotype {
        (self.graph >> i & 1) as Genotype
    }

    #[inline]
    pub fn set_row(&mut self, i: usize, genotype: i8) {
        match genotype {
            0 => self.graph &= !(1 << i),
            1 => self.graph |= 1 << i,
            _ => panic!("Invalid genotype"),
        };
    }
    #[inline]
    pub fn is_het(&self) -> bool {
        self.meta.get_segment_marker()
    }

    #[inline]
    pub fn is_segment_marker(&self) -> bool {
        self.meta.get_segment_marker()
    }

    #[inline]
    pub fn unset_segment_marker(&mut self) {
        self.meta.unset_segment_marker()
    }

    #[inline]
    pub fn get_genotype(&self) -> Genotype {
        match self.graph {
            0b0000000 => 0,
            0b1111111 => 2,
            _ => 1,
        }
    }
}

pub struct GenotypeGraph {
    pub graph: Array1<G>,
}

impl GenotypeGraph {
    pub fn build(genotypes: ArrayView1<Genotype>) -> Self {
        let m = genotypes.len();

        let mut graph = Vec::with_capacity(m);

        let mut cur_het_count = tp_value!(HET_PER_SEGMENT - 1, u32);
        let mut first = true;

        for &g in genotypes {
            if g == 1 {
                cur_het_count += 1;
                if cur_het_count == HET_PER_SEGMENT as u32 {
                    cur_het_count = 0;
                }
                graph.push(G::new_het(cur_het_count));
                if first {
                    graph.last_mut().unwrap().unset_segment_marker();
                    first = false;
                }
            } else {
                graph.push(G::new_hom(g));
            }
        }

        Self {
            graph: Array1::from_vec(graph),
        }
    }

    pub fn slice<'a>(&'a mut self, start: usize, end: usize) -> GenotypeGraphSlice<'a> {
        GenotypeGraphSlice {
            graph: self.graph.slice(s![start..end]),
        }
    }

    // TODO: limit merges to maximum of MAX_AMBIGUOUS het sites within a block
    pub fn prune(&mut self, tprob: ArrayView3<Real>) {
        #[cfg(feature = "leak-resist")]
        let m = tprob.shape()[0];

        #[cfg(not(feature = "leak-resist"))]
        let m = tprob.shape()[0];

        // Backward pass to identify where to merge adjacent blocks and the new haps
        let mut ind_cache = unsafe { Array3::<U8>::uninit((m, P, 2)).assume_init() };
        let mut merge_head = vec![tp_value!(false, bool); m];
        let mut merge_flag = tp_value!(false, bool);
        let mut new_merge_flag;

        for i in (0..m - 1).rev() {
            let (ind, prob, _) = select_top_p(tprob.slice(s![i, .., ..]));
            // If merge flag set then carry over
            for j in 0..P {
                #[cfg(feature = "leak-resist")]
                {
                    ind1_cache[[i, j]] = merge_flag.select(ind1_cache[[i + 1, j]], ind1[j]);
                    ind2_cache[[i, j]] = merge_flag.select(ind2_cache[[i + 1, j]], ind2[j]);
                }

                #[cfg(not(feature = "leak-resist"))]
                {
                    ind_cache[[i, j, 0]] = if merge_flag {
                        ind_cache[[i + 1, j, 0]]
                    } else {
                        ind[[j, 0]]
                    };
                    ind_cache[[i, j, 1]] = if merge_flag {
                        ind_cache[[i + 1, j, 1]]
                    } else {
                        ind[[j, 1]]
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
        //let mut new_geno = unsafe { Array1::<Genotype>::uninit(P).assume_init() };
        let mut ind = unsafe { Array2::<U8>::uninit((P, 2)).assume_init() };
        let mut ind_final = unsafe { Array1::<U8>::uninit(P).assume_init() };
        let mut block_counter = tp_value!(0, i8);

        for i in 0..m {
            let mut new_geno = self.graph[i];
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
                        ind[[j, 0]] = ind_cache[[i, j, 0]];
                        ind[[j, 1]] = ind_cache[[i, j, 1]];
                    }

                    // If block_counter hits zero change back to original indicies
                    if block_counter == 0 {
                        ind[[j, 0]] = j as u8;
                        ind[[j, 1]] = j as u8;
                    }

                    // Use ind1 for first block, ind2 for second block
                    if block_counter == 2 {
                        ind_final[j] = ind[[j, 0]];
                    } else {
                        ind_final[j] = ind[[j, 1]];
                    }
                }

                for j in 0..P {
                    new_geno.set_row(j, self.graph[i].get_row(ind_final[j] as usize));
                }

                // Erase block_head flag between the two blocks being merged
                if self.graph[i].is_segment_marker() && block_counter == 1 {
                    new_geno.unset_segment_marker();
                }

                self.graph[i] = new_geno;
            }
        }
    }

    pub fn prune_rank(&mut self, tprobs: ArrayView3<Real>) {
        let mut trans_map = Vec::new();
        let mut het_count = 0;
        let mut segment_start_i = 0;
        for (i, g) in self.graph.iter().enumerate() {
            if g.is_segment_marker() {
                trans_map.push((segment_start_i, het_count));
                segment_start_i = i;
                het_count = 0;
            }
            het_count += g.is_het() as usize;
        }
        trans_map.push((segment_start_i, het_count));

        let mut trans_stats = Vec::new();
        for (i, (t1, t2)) in trans_map.iter().zip(trans_map.iter().skip(1)).enumerate() {
            let het_count = t1.1 + t2.1;
            if het_count < MAX_HETS {
                let (ind, prob, entrophy) = select_top_p(tprobs.slice(s![t2.0, .., ..]));
                if prob > MCMC_PRUNE_PROB_THRES {
                    trans_stats.push((entrophy, i, ind));
                }
            }
        }

        trans_stats.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut trans_ind = vec![None; trans_map.len() - 1];

        for t in trans_stats {
            let (_, i, ind) = t;
            trans_ind[i] = if i == 0 && i == trans_ind.len() - 1 {
                None
            } else if i == 0 && i < trans_ind.len() - 1 {
                if trans_ind[i + 1].is_none() {
                    Some(ind)
                } else {
                    None
                }
            } else if i > 0 && i == trans_ind.len() - 1 {
                if trans_ind[i - 1].is_none() {
                    Some(ind)
                } else {
                    None
                }
            } else {
                if trans_ind[i - 1].is_none() && trans_ind[i + 1].is_none() {
                    Some(ind)
                } else {
                    None
                }
            }
        }

        let mut trans_ind_iter = trans_ind.iter();
        let mut cur_trans_ind = &None;

        // forward scan
        for g in &mut self.graph.iter_mut() {
            if g.is_segment_marker() {
                cur_trans_ind = trans_ind_iter.next().unwrap();
            }
            if let Some(cur_trans_ind) = cur_trans_ind {
                let mut new_g = *g;
                Zip::indexed(cur_trans_ind.rows()).for_each(|i, ind| {
                    new_g.set_row(i as usize, g.get_row(ind[1] as usize));
                });
                *g = new_g;
            }
        }
        // backward scan
        let mut trans_ind_iter = trans_ind.iter().rev();
        let mut cur_trans_ind: &Option<Array2<U8>> = &None;
        for g in &mut self.graph.iter_mut().rev() {
            if let Some(cur_trans_ind) = cur_trans_ind {
                let mut new_g = *g;
                Zip::indexed(cur_trans_ind.rows()).for_each(|i, ind| {
                    new_g.set_row(i as usize, g.get_row(ind[0] as usize));
                });
                *g = new_g;
            }
            if g.is_segment_marker() {
                cur_trans_ind = trans_ind_iter.next().unwrap();
                if cur_trans_ind.is_some() {
                    g.unset_segment_marker();
                }
            }
        }
    }

    pub fn traverse_graph_pair(&self, ind: ArrayView2<u8>, mut haps: ArrayViewMut2<Genotype>) {
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

fn select_top_p(tab: ArrayView2<Real>) -> (Array2<U8>, Real, Real) {
    let n = P * P;
    let mut elems = Vec::with_capacity(n);
    let mut entrophy = 0.;
    for i in 0..P {
        for j in 0..P {
            //let prob = tab[[i, j]] * tab[[P - 1 - i, P - 1 - j]];
            let prob = tab[[i, j]];
            entrophy += if prob == 0. { 0. } else { -prob * prob.log10() };
            #[cfg(feature = "leak-resist")]
            {
                elems.push(SortItem {
                    prob,
                    i: U8::protect(i as u8),
                    j: U8::protect(j as u8),
                });
            }

            #[cfg(not(feature = "leak-resist"))]
            {
                elems.push((prob, i as u8, j as u8));
            }
        }
    }

    // Descending sort
    #[cfg(feature = "leak-resist")]
    {
        oram_sgx::BiotonicSort::sort(&mut elems[..], false);
    }

    #[cfg(not(feature = "leak-resist"))]
    {
        elems.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
    }

    let mut ind = unsafe { Array2::<U8>::uninit((P, 2)).assume_init() };

    let mut taken = Array2::<bool>::from_elem((P, P), false);

    let mut sum = 0.;
    let mut count = 0;

    for e in elems {
        sum += e.0;
        let (i, j) = (e.1 as usize, e.2 as usize);
        if !taken[[i, j]] && !taken[[P - 1 - i, P - 1 - j]] {
            ind[[count, 0]] = i as u8;
            ind[[count, 1]] = j as u8;
            ind[[P - 1 - count, 0]] = (P - 1 - i) as u8;
            ind[[P - 1 - count, 1]] = (P - 1 - j) as u8;
            count += 1;
            taken[[i, j]] = true;
            taken[[P - 1 - i, P - 1 - j]] = true;
        }

        if count == P / 2 {
            break;
        }
    }
    (ind, sum, entrophy)
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
