use crate::{Bool, Genotype, U8};
use ndarray::{s, Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut2, Zip};

pub const HET_PER_SEGMENT: u8 = 3;
pub const P: usize = 1 << HET_PER_SEGMENT;
pub const MCMC_PRUNE_PROB_THRES: f64 = 0.999; // "mcmc-prune" parameter in ShapeIt4
pub const MAX_HETS: usize = 22; // "MAX_AMB" constant in ShapeIt4 (utils/otools.h)

#[cfg(feature = "obliv")]
use tp_fixedpoint::timing_shield::{TpCondSwap, TpEq, TpOrd};

//#[cfg(feature = "obliv")]
//type Real = crate::RealHmm;

//#[cfg(not(feature = "obliv"))]
type Real = f64;

//#[cfg(feature = "obliv")]
//mod inner {
//use super::*;
//pub use crate::oram::SmallLSOram;
//pub use tp_fixedpoint::timing_shield::{TpBool, TpCondSwap, TpEq, TpOrd};
//pub struct SortItem {
//pub dip: Real,
//pub i: U8,
//pub j: U8,
//}

//impl TpOrd for SortItem {
//fn tp_lt(&self, rhs: &Self) -> TpBool {
//self.dip.tp_lt(&rhs.dip)
//}

//fn tp_lt_eq(&self, rhs: &Self) -> TpBool {
//self.dip.tp_lt_eq(&rhs.dip)
//}

//fn tp_gt(&self, rhs: &Self) -> TpBool {
//self.dip.tp_gt(&rhs.dip)
//}

//fn tp_gt_eq(&self, rhs: &Self) -> TpBool {
//self.dip.tp_gt_eq(&rhs.dip)
//}
//}

//impl TpCondSwap for SortItem {
//fn tp_cond_swap(cond: TpBool, a: &mut Self, b: &mut Self) {
//Real::tp_cond_swap(cond, &mut a.dip, &mut b.dip);
//U8::tp_cond_swap(cond, &mut a.i, &mut b.i);
//U8::tp_cond_swap(cond, &mut a.j, &mut b.j);
//}
//}
//}

//#[cfg(feature = "obliv")]
//use inner::*;

#[derive(Clone, Copy)]
pub struct GMeta(U8);

const SEGMENT_MARKER_MASK: u8 = 0b11111110;

impl GMeta {
    fn new(is_segment_marker: Bool, is_het: Bool, is_alt_allele: Bool) -> Self {
        #[cfg(feature = "obliv")]
        let mut inner = U8::protect(0);
        #[cfg(feature = "obliv")]
        {
            inner |= is_segment_marker.as_u8();
            inner |= (is_het.as_u8()) << 1;
            inner |= (is_alt_allele.as_u8()) << 2;
        }

        #[cfg(not(feature = "obliv"))]
        let mut inner = 0u8;

        #[cfg(not(feature = "obliv"))]
        {
            inner |= is_segment_marker as u8;
            inner |= (is_het as u8) << 1;
            inner |= (is_alt_allele as u8) << 2;
        }
        Self(inner)
    }

    #[inline]
    fn get_segment_marker(self) -> Bool {
        #[cfg(feature = "obliv")]
        return (self.0 & 1).tp_eq(&1);

        #[cfg(not(feature = "obliv"))]
        return self.0 & 1 == 1;
    }

    #[inline]
    fn get_het(self) -> Bool {
        #[cfg(feature = "obliv")]
        return ((self.0 >> 1) & 1).tp_eq(&1);

        #[cfg(not(feature = "obliv"))]
        return ((self.0 >> 1) & 1) == 1;
    }

    #[inline]
    fn unset_segment_marker(&mut self, #[cfg(feature = "obliv")] cond: Bool) {
        #[cfg(feature = "obliv")]
        {
            self.0 = cond.select(self.0 & SEGMENT_MARKER_MASK, self.0);
        }

        #[cfg(not(feature = "obliv"))]
        {
            self.0 &= SEGMENT_MARKER_MASK
        }
    }
}

#[cfg(feature = "obliv")]
impl TpCondSwap for GMeta {
    fn tp_cond_swap(cond: Bool, a: &mut Self, b: &mut Self) {
        U8::tp_cond_swap(cond, &mut a.0, &mut b.0);
    }
}

#[derive(Copy, Clone)]
pub struct G {
    graph: U8,
    meta: GMeta,
}

impl G {
    pub fn new_het(het_count: U8) -> Self {
        #[cfg(feature = "obliv")]
        let graph = het_count.tp_eq(&0).select(
            U8::protect(0b01010101),
            het_count
                .tp_eq(&1)
                .select(U8::protect(0b00110011), U8::protect(0b00001111)),
        );

        #[cfg(feature = "obliv")]
        let segment_marker = het_count.tp_eq(&0);

        #[cfg(not(feature = "obliv"))]
        let (graph, segment_marker) = match het_count {
            0 => (0b01010101, true),
            1 => (0b00110011, false),
            2 => (0b00001111, false),
            _ => panic!("Invalid het count"),
        };
        Self {
            graph,
            #[cfg(feature = "obliv")]
            meta: GMeta::new(segment_marker, Bool::protect(true), Bool::protect(false)),
            #[cfg(not(feature = "obliv"))]
            meta: GMeta::new(segment_marker, true, false),
        }
    }

    pub fn new_hom(hom: Genotype) -> Self {
        #[cfg(feature = "obliv")]
        let graph = hom
            .tp_eq(&0)
            .select(U8::protect(0b00000000), U8::protect(0b11111111));

        #[cfg(not(feature = "obliv"))]
        let graph = match hom {
            0 => 0b00000000,
            2 => 0b11111111,
            _ => panic!("Invalid homozygote"),
        };
        Self {
            graph,
            #[cfg(feature = "obliv")]
            meta: GMeta::new(Bool::protect(false), Bool::protect(false), hom.tp_eq(&2)),
            #[cfg(not(feature = "obliv"))]
            meta: GMeta::new(false, false, hom == 2),
        }
    }

    #[inline]
    pub fn get_row(&self, i: usize) -> Genotype {
        #[cfg(feature = "obliv")]
        return (self.graph >> i as u32 & 1).as_i8();

        #[cfg(not(feature = "obliv"))]
        return (self.graph >> i & 1) as Genotype;
    }

    #[inline]
    #[cfg(feature = "obliv")]
    pub fn get_row_obliv(&self, i: U8) -> Genotype {
        (self.graph >> i.as_u32().expose() & 1).as_i8()
    }

    #[inline]
    pub fn set_row(&mut self, i: usize, genotype: Genotype) {
        #[cfg(feature = "obliv")]
        {
            self.graph = genotype
                .tp_eq(&0)
                .select(self.graph & !(1 << i), self.graph | 1 << i);
        }

        #[cfg(not(feature = "obliv"))]
        match genotype {
            0 => self.graph &= !(1 << i),
            1 => self.graph |= 1 << i,
            _ => panic!("Invalid genotype"),
        };
    }
    #[inline]
    pub fn is_het(&self) -> Bool {
        self.meta.get_segment_marker()
    }

    #[inline]
    pub fn is_segment_marker(&self) -> Bool {
        self.meta.get_segment_marker()
    }

    #[inline]
    fn unset_segment_marker(&mut self, #[cfg(feature = "obliv")] cond: Bool) {
        #[cfg(feature = "obliv")]
        {
            self.meta.unset_segment_marker(cond);
        }
        #[cfg(not(feature = "obliv"))]
        {
            self.meta.unset_segment_marker();
        }
    }

    #[inline]
    pub fn get_genotype(&self) -> Genotype {
        #[cfg(feature = "obliv")]
        return self.graph.tp_eq(&0b0000000).select(
            Genotype::protect(0),
            self.graph
                .tp_eq(&0b1111111)
                .select(Genotype::protect(2), Genotype::protect(1)),
        );

        #[cfg(not(feature = "obliv"))]
        match self.graph {
            0b0000000 => 0,
            0b1111111 => 2,
            _ => 1,
        }
    }
}

#[cfg(feature = "obliv")]
impl TpCondSwap for G {
    fn tp_cond_swap(cond: Bool, a: &mut Self, b: &mut Self) {
        U8::tp_cond_swap(cond, &mut a.graph, &mut b.graph);
        GMeta::tp_cond_swap(cond, &mut a.meta, &mut b.meta);
    }
}

pub struct GenotypeGraph {
    pub graph: Array1<G>,
}

impl GenotypeGraph {
    pub fn build(genotypes: ArrayView1<Genotype>) -> Self {
        let m = genotypes.len();

        let mut graph = Vec::with_capacity(m);

        #[cfg(feature = "obliv")]
        let mut cur_het_count = U8::protect(HET_PER_SEGMENT - 1);
        #[cfg(not(feature = "obliv"))]
        let mut cur_het_count = HET_PER_SEGMENT - 1;

        #[cfg(feature = "obliv")]
        let mut first = Bool::protect(true);

        #[cfg(not(feature = "obliv"))]
        let mut first = true;

        for &g in genotypes {
            #[cfg(feature = "obliv")]
            {
                let cond = g.tp_eq(&1);
                cur_het_count = cond.select(cur_het_count + 1, cur_het_count);
                cur_het_count = cur_het_count
                    .tp_eq(&HET_PER_SEGMENT)
                    .select(U8::protect(0), cur_het_count);
                let new_graph = cond.select(G::new_het(cur_het_count), G::new_hom(g));
                graph.push(new_graph);
                graph.last_mut().unwrap().unset_segment_marker(first);
                first = cond.select(Bool::protect(false), first);
            }

            #[cfg(not(feature = "obliv"))]
            if g == 1 {
                cur_het_count += 1;
                if cur_het_count == HET_PER_SEGMENT {
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

    pub fn slice<'a>(&'a self, start: usize, end: usize) -> GenotypeGraphSlice<'a> {
        GenotypeGraphSlice {
            graph: self.graph.slice(s![start..end]),
        }
    }

    // TODO: limit merges to maximum of MAX_AMBIGUOUS het sites within a block
    pub fn prune(&mut self, tprob: ArrayView3<Real>) {
        #[cfg(feature = "obliv")]
        let m = tprob.shape()[0];

        #[cfg(not(feature = "obliv"))]
        let m = tprob.shape()[0];

        // Backward pass to identify where to merge adjacent blocks and the new haps
        #[cfg(feature = "obliv")]
        let mut ind_cache = Array3::<U8>::from_elem((m, P, 2), U8::protect(0));

        #[cfg(feature = "obliv")]
        let mut merge_head = vec![Bool::protect(false); m];

        #[cfg(feature = "obliv")]
        let mut merge_flag = Bool::protect(false);

        #[cfg(not(feature = "obliv"))]
        let mut ind_cache = Array3::<u8>::zeros((m, P, 2));

        #[cfg(not(feature = "obliv"))]
        let mut merge_head = vec![false; m];

        #[cfg(not(feature = "obliv"))]
        let mut merge_flag = false;

        let mut new_merge_flag;

        for i in (0..m - 1).rev() {
            #[cfg(feature = "obliv")]
            let (ind, prob) = {
                //let tprob = tprob
                    //.slice(s![i, .., ..])
                    //.map(|v| v.expose_into_f32() as f64);
                //let (ind, prob, _) = select_top_p(tprob.view());
                let (ind, prob) = select_top_p(tprob.slice(s![i, .., ..]));
                (ind.map(|&v| U8::protect(v)), prob as f32)
            };

            #[cfg(not(feature = "obliv"))]
            let (ind, prob) = select_top_p(tprob.slice(s![i, .., ..]));

            // If merge flag set then carry over
            for j in 0..P {
                #[cfg(feature = "obliv")]
                {
                    ind_cache[[i, j, 0]] = merge_flag.select(ind_cache[[i + 1, j, 0]], ind[[j, 0]]);
                    ind_cache[[i, j, 1]] = merge_flag.select(ind_cache[[i + 1, j, 1]], ind[[j, 1]]);
                }

                #[cfg(not(feature = "obliv"))]
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

            #[cfg(feature = "obliv")]
            {
                let is_segment_marker = self.graph[i].is_segment_marker();
                // If at head and merge is on, it is the merge head
                merge_head[i] = ((i == 0) | is_segment_marker) & merge_flag;

                // If merge is not on, we're at a head and prob over threshold, start a new merge
                new_merge_flag = is_segment_marker
                    //& (prob.tp_gt(&Real::protect_f32(MCMC_PRUNE_PROB_THRES as f32)))
                    & (prob > MCMC_PRUNE_PROB_THRES as f32 )
                    & !merge_flag;

                // If at head then merge_flag is set to new_merge_flag, otherwise carry over
                merge_flag = is_segment_marker.select(new_merge_flag, merge_flag);
            }

            #[cfg(not(feature = "obliv"))]
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
        #[cfg(feature = "obliv")]
        let mut ind = Array2::<U8>::from_elem((P, 2), U8::protect(0));
        #[cfg(feature = "obliv")]
        let mut ind_final = Array1::<U8>::from_elem(P, U8::protect(0));
        #[cfg(feature = "obliv")]
        let mut block_counter = U8::protect(0);

        #[cfg(not(feature = "obliv"))]
        let mut ind = Array2::<U8>::zeros((P, 2));
        #[cfg(not(feature = "obliv"))]
        let mut ind_final = Array1::<U8>::zeros(P);

        #[cfg(not(feature = "obliv"))]
        let mut block_counter = 0u8;

        for i in 0..m {
            let mut new_geno = self.graph[i];
            #[cfg(feature = "obliv")]
            {
                let is_segment_marker = self.graph[i].is_segment_marker();
                // At merge head set block counter to 2 (will update the next two blocks)
                block_counter = merge_head[i].select(U8::protect(2), block_counter);
                // Every time we see a block head that is not a merge head, decrement block counter
                block_counter = (!merge_head[i] & is_segment_marker).select(
                    {
                        let v = (block_counter.as_i8() - 1).as_u8();
                        v.tp_gt(&0).select(v, U8::protect(0))
                    },
                    block_counter,
                );

                for j in 0..P {
                    // If at merge head copy cache over
                    ind[[j, 0]] = merge_head[i].select(ind_cache[[i, j, 0]], ind[[j, 0]]);
                    ind[[j, 1]] = merge_head[i].select(ind_cache[[i, j, 1]], ind[[j, 1]]);

                    // If block_counter hits zero change back to original indicies
                    let cond = block_counter.tp_eq(&0);
                    ind[[j, 0]] = cond.select(U8::protect(j as u8), ind[[j, 0]]);
                    ind[[j, 1]] = cond.select(U8::protect(j as u8), ind[[j, 1]]);

                    // Use ind1 for first block, ind2 for second block
                    ind_final[j] = block_counter.tp_eq(&2).select(ind[[j, 0]], ind[[j, 1]]);
                }

                for j in 0..P {
                    new_geno.set_row(j, self.graph[i].get_row_obliv(ind_final[j]));
                }

                // Erase block_head flag between the two blocks being merged
                new_geno.unset_segment_marker(
                    self.graph[i].is_segment_marker() & block_counter.tp_eq(&1),
                );

                self.graph[i] = new_geno;
            }

            #[cfg(not(feature = "obliv"))]
            {
                let is_segment_marker = self.graph[i].is_segment_marker();
                // At merge head set block counter to 2 (will update the next two blocks)
                if merge_head[i] {
                    block_counter = 2;
                }
                // Every time we see a block head that is not a merge head, decrement block counter
                if !merge_head[i] && is_segment_marker {
                    block_counter = 0.max((block_counter as i8) - 1) as u8;
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

    #[cfg(not(feature = "obliv"))]
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
                let (ind, prob, entrophy) = select_top_p_with_entropy(tprobs.slice(s![t2.0, .., ..]));
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

    pub fn traverse_graph_pair(&self, ind: ArrayView2<U8>, mut haps: ArrayViewMut2<Genotype>) {
        Zip::from(haps.rows_mut())
            .and(ind.rows())
            .and(&self.graph)
            .for_each(|mut h_row, ind_row, g| {
                #[cfg(feature = "obliv")]
                {
                    h_row[0] = g.get_row_obliv(ind_row[0]);
                    h_row[1] = g.get_row_obliv(ind_row[1]);
                }
                #[cfg(not(feature = "obliv"))]
                {
                    h_row[0] = g.get_row(ind_row[0] as usize);
                    h_row[1] = g.get_row(ind_row[1] as usize);
                }
            });
    }
}

pub struct GenotypeGraphSlice<'a> {
    pub graph: ArrayView1<'a, G>,
}

fn select_top_p(tab: ArrayView2<Real>) -> (Array2<u8>, f64) {
    //#[cfg(feature = "obliv")]
    //let tab = tab.map(|v| v.expose_into_f32() as f64);

    let n = P * P;
    let mut elems = Vec::with_capacity(n);
    for i in 0..P {
        for j in 0..P {
            elems.push((tab[[i, j]], i as u8, j as u8));
        }
    }

    elems.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());

    let mut ind = Array2::<u8>::zeros((P, 2));

    let mut taken = Array2::<bool>::from_elem((P, P), false);

    let mut sum = 0.;
    let mut count = 0;

    for e in elems.clone() {
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
    (ind, sum)
}

fn select_top_p_with_entropy(tab: ArrayView2<Real>) -> (Array2<u8>, f64, f64) {
    //#[cfg(feature = "obliv")]
    //let tab = tab.map(|v| v.expose_into_f32() as f64);

    let n = P * P;
    let mut elems = Vec::with_capacity(n);
    let mut entrophy = 0.;
    for i in 0..P {
        for j in 0..P {
            let prob = tab[[i, j]];
            entrophy += if prob == 0. { 0. } else { -prob * prob.log10() };
            elems.push((prob, i as u8, j as u8));
        }
    }

    elems.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());

    let mut ind = Array2::<u8>::zeros((P, 2));

    let mut taken = Array2::<bool>::from_elem((P, P), false);

    let mut sum = 0.;
    let mut count = 0;

    for e in elems.clone() {
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

//fn select_top_p(tab: ArrayView2<Real>) -> (Array2<U8>, Real, Real) {
//let n = P * P;
//let mut elems = Vec::with_capacity(n);
//let mut entrophy = 0.;
//for i in 0..P {
//for j in 0..P {
//let prob = tab[[i, j]];
//entrophy += if prob == 0. { 0. } else { -prob * prob.log10() };
//#[cfg(feature = "obliv")]
//{
//elems.push(SortItem {
//prob,
//i: U8::protect(i as u8),
//j: U8::protect(j as u8),
//});
//}

//#[cfg(not(feature = "obliv"))]
//{
//elems.push((prob, i as u8, j as u8));
//}
//}
//}

//// Descending sort
//#[cfg(feature = "obliv")]
//{
//oram_sgx::BiotonicSort::sort(&mut elems[..], false);
//}

//#[cfg(not(feature = "obliv"))]
//{
//elems.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
//}

//let mut ind = Array2::<U8>::zeros((P, 2));

//let mut taken = Array2::<bool>::from_elem((P, P), false);

//let mut sum = 0.;
//let mut count = 0;

//for e in elems.clone() {
//sum += e.0;
//let (i, j) = (e.1 as usize, e.2 as usize);
//if !taken[[i, j]] && !taken[[P - 1 - i, P - 1 - j]] {
//ind[[count, 0]] = i as u8;
//ind[[count, 1]] = j as u8;
//ind[[P - 1 - count, 0]] = (P - 1 - i) as u8;
//ind[[P - 1 - count, 1]] = (P - 1 - j) as u8;
//count += 1;
//taken[[i, j]] = true;
//taken[[P - 1 - i, P - 1 - j]] = true;
//}

//if count == P / 2 {
//break;
//}
//}
//(ind, sum, entrophy)
//}
