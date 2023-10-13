use bitvec::prelude::{BitSlice, BitVec, Lsb0};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, Zip};
use std::collections::HashMap;

pub struct Block {
    pub index_map: Array1<usize>,
    pub haplotypes: Array2<u8>,
    //pub weights: Array1<f64>,
    //pub inv_weights: Array1<f64>,
    pub n_unique: usize,
}

impl Block {
    pub fn n_sites(&self) -> usize {
        self.haplotypes.nrows()
    }

    pub fn n_full(&self) -> usize {
        self.index_map.len()
    }

    pub fn n_unique(&self) -> usize {
        self.n_unique
    }

    pub fn as_slice<'a>(&'a self) -> BlockSlice<'a> {
        BlockSlice {
            index_map: self.index_map.view(),
            haplotypes: self.haplotypes.view(),
            //weights: self.weights.view(),
            //inv_weights: self.inv_weights.view(),
            n_unique: self.n_unique,
        }
    }

    pub fn slice<'a>(&'a self, start: usize, end: usize) -> BlockSlice<'a> {
        BlockSlice {
            index_map: self.index_map.view(),
            haplotypes: self.haplotypes.slice(s![start..end, ..]),
            //weights: self.weights.view(),
            //inv_weights: self.inv_weights.view(),
            n_unique: self.n_unique,
        }
    }

    pub fn expand_pos(&self, pos: usize) -> Array1<i8> {
        let haps = self.haplotypes.row(pos);
        self.expand(haps)
    }

    fn expand(&self, haps: ArrayView1<u8>) -> Array1<i8> {
        let haps = BitSlice::<u8, Lsb0>::from_slice(haps.as_slice().unwrap());
        let mut expanded = Array1::<i8>::zeros(self.n_unique());
        for (src, tar) in haps.iter().zip(expanded.iter_mut()) {
            *tar = *src as i8;
        }
        expanded
    }

    pub fn transpose(&self) -> TransposedBlock {
        let ncols = (self.n_sites() + 7) / 8;
        let mut transposed_haplotypes = Array2::<u8>::zeros((self.n_unique(), ncols));
        Zip::indexed(self.haplotypes.rows()).for_each(|i, s| {
            let s = BitSlice::<u8, Lsb0>::from_slice(s.as_slice().unwrap());
            for (b, mut hap) in s.iter().zip(transposed_haplotypes.rows_mut().into_iter()) {
                let hap = BitSlice::<u8, Lsb0>::from_slice_mut(hap.as_slice_mut().unwrap());
                hap.set(i, *b);
            }
        });

        TransposedBlock {
            index_map: self.index_map.clone(),
            haplotypes: transposed_haplotypes,
            n_sites: self.n_sites(),
        }
    }
}

fn transpose_haps(haps: ArrayView2<u8>, ncols: usize) -> Array2<u8> {
    assert_eq!(haps.ncols(), (ncols + 7) / 8);
    let haps_rows = haps.rows().into_iter().collect::<Vec<_>>();
    let haps_bits = haps_rows
        .iter()
        .map(|r| BitSlice::<u8, Lsb0>::from_slice(r.as_slice().unwrap()));

    let mut transposed_haps = Array2::<u8>::zeros((ncols, (haps.nrows() + 7) / 8));
    let mut transposed_haps_rows = transposed_haps.rows_mut().into_iter().collect::<Vec<_>>();
    let mut transposed_haps_bits = transposed_haps_rows
        .iter_mut()
        .map(|r| BitSlice::<u8, Lsb0>::from_slice_mut(r.as_slice_mut().unwrap()))
        .collect::<Vec<_>>();

    for (i, h) in haps_bits.enumerate() {
        for (t, b) in transposed_haps_bits.iter_mut().zip(h.iter()) {
            t.set(i, *b);
        }
    }

    transposed_haps
}
pub fn make_unique_hap_block(block: &Block) -> Block {
    let block_transposed = block.transpose();
    let mut unique = HashMap::new();
    let mut new_index_map = Array1::<usize>::zeros(block.index_map.len());
    Zip::from(&block.index_map)
        .and(&mut new_index_map)
        .for_each(|&i_b, i| {
            let item = block_transposed.haplotypes.row(i_b);
            let cur_len = unique.len();
            let c = unique.entry(item.clone()).or_insert(cur_len);
            *i = *c;
        });
    let n_unique = unique.len();
    //let mut weights = Array1::<f64>::zeros(n_unique);
    //for &i in &new_index_map {
    //weights[i as usize] += 1.;
    //}

    let mut new_haps = Array2::<u8>::zeros((n_unique, block.n_sites().div_ceil(8)));
    for (b, i) in unique.into_iter() {
        let bv = BitVec::<u8, Lsb0>::from_slice(b.as_slice().unwrap());
        new_haps
            .row_mut(i)
            .assign(&ArrayView1::<u8>::from(bv.as_raw_slice()));
    }

    let new_haps_transposed = transpose_haps(new_haps.view(), block.n_sites());
    Block {
        index_map: new_index_map,
        haplotypes: new_haps_transposed,
        //inv_weights: weights.map(|v| 1. / v),
        //weights,
        n_unique,
    }
}

pub fn merge_blocks(left_block: &Block, right_block: &Block) -> Block {
    let left_transposed = left_block.transpose();
    let right_transposed = right_block.transpose();

    let mut unique = HashMap::new();
    let mut new_index_map = Array1::<usize>::zeros(left_block.index_map.len());

    Zip::from(&left_block.index_map)
        .and(&right_block.index_map)
        .and(&mut new_index_map)
        .for_each(|&i_l, &i_r, i| {
            let item = (
                left_transposed.haplotypes.row(i_l),
                right_transposed.haplotypes.row(i_r),
            );
            let cur_len = unique.len();
            let c = unique.entry(item.clone()).or_insert(cur_len);
            *i = *c;
        });

    let n_unique = unique.len();
    //let mut weights = Array1::<f64>::zeros(n_unique);
    //for &i in &new_index_map {
    //weights[i as usize] += 1.;
    //}

    let mut new_haps = Array2::<u8>::zeros((
        n_unique,
        ((left_block.n_sites() + right_block.n_sites()) + 7) / 8,
    ));

    for ((left, right), i) in unique.into_iter() {
        let mut bv = BitVec::<u8, Lsb0>::from_slice(left.as_slice().unwrap());
        bv.truncate(left_block.n_sites());
        let right_bv = BitSlice::<u8, Lsb0>::from_slice(right.as_slice().unwrap());
        bv.extend_from_bitslice(&right_bv);
        bv.truncate(left_block.n_sites() + right_block.n_sites());
        new_haps
            .row_mut(i)
            .assign(&ArrayView1::<u8>::from(bv.as_raw_slice()));
    }

    let new_haps_transposed = transpose_haps(
        new_haps.view(),
        left_block.n_sites() + right_block.n_sites(),
    );
    Block {
        index_map: new_index_map,
        haplotypes: new_haps_transposed,
        //inv_weights: weights.map(|v| 1. / v),
        //weights,
        n_unique,
    }
}

#[derive(Clone)]
pub struct BlockSlice<'a> {
    pub index_map: ArrayView1<'a, usize>,
    pub haplotypes: ArrayView2<'a, u8>,
    //pub weights: ArrayView1<'a, f64>,
    //pub inv_weights: ArrayView1<'a, f64>,
    pub n_unique: usize,
}

impl<'a> BlockSlice<'a> {
    pub fn n_sites(&self) -> usize {
        self.haplotypes.nrows()
    }

    pub fn n_full(&self) -> usize {
        self.index_map.len()
    }

    pub fn n_unique(&self) -> usize {
        self.n_unique
    }

    pub fn iter(&self) -> impl Iterator<Item = Vec<bool>> + '_ {
        self.haplotypes.rows().into_iter().map(|v| {
            BitSlice::<u8, Lsb0>::from_slice(v.as_slice().unwrap())
                .iter()
                .map(|v| *v)
                .collect::<Vec<_>>()
        })
    }

    pub fn expand_pos(&self, pos: usize) -> Array1<i8> {
        let haps = self.haplotypes.row(pos);
        self.expand(haps)
    }

    fn expand(&self, haps: ArrayView1<u8>) -> Array1<i8> {
        let haps = BitSlice::<u8, Lsb0>::from_slice(haps.as_slice().unwrap());
        let mut expanded = Array1::<i8>::zeros(self.n_unique());
        for (src, tar) in haps.iter().zip(expanded.iter_mut()) {
            *tar = *src as i8;
        }
        expanded
    }

    pub fn get_members(&self) -> Vec<Vec<usize>> {
        let mut members = vec![Vec::new(); self.n_unique];
        for (i, &index) in self.index_map.iter().enumerate() {
            members[index].push(i);
        }
        members
    }
}

pub struct TransposedBlock {
    pub index_map: Array1<usize>,
    pub haplotypes: Array2<u8>,
    pub n_sites: usize,
}

impl TransposedBlock {
    pub fn slice(&self, start: usize, end: usize) -> TransposedBlockSlice {
        TransposedBlockSlice {
            index_map: self.index_map.view(),
            haplotypes: self.haplotypes.view(),
            n_sites: end - start,
            range: Some((start, end)),
        }
    }

    pub fn as_slice(&self) -> TransposedBlockSlice {
        TransposedBlockSlice {
            index_map: self.index_map.view(),
            haplotypes: self.haplotypes.view(),
            n_sites: self.n_sites,
            range: None,
        }
    }
}

pub struct TransposedBlockSlice<'a> {
    index_map: ArrayView1<'a, usize>,
    haplotypes: ArrayView2<'a, u8>,
    n_sites: usize,
    range: Option<(usize, usize)>,
}

impl<'a> TransposedBlockSlice<'a> {
    pub fn filter(&self, bitmask: &[bool], mut output: ArrayViewMut2<i8>) {
        assert_eq!(output.nrows(), self.n_sites);
        let mut hap_count = 0;
        for (i, b) in bitmask.iter().enumerate() {
            if *b {
                let hap_i = self.index_map[i] as usize;
                let tmp = self.haplotypes.row(hap_i);
                let hap = BitSlice::<u8, Lsb0>::from_slice(tmp.as_slice().unwrap());
                let hap = if let Some(&(start, end)) = self.range.as_ref() {
                    &hap[start..end]
                } else {
                    &hap[..self.n_sites]
                };
                for (mut p, b) in output.rows_mut().into_iter().zip(hap.iter()) {
                    p[hap_count] = *b as i8;
                }
                hap_count += 1;
            }
        }
    }

    pub fn n_sites(&self) -> usize {
        self.n_sites
    }
}

pub struct ExpandedBlock {
    pub rhap: Array2<i8>,
}

impl ExpandedBlock {
    pub fn from_block_slice<'a>(block: BlockSlice<'a>) -> Self {
        let mut rhap = Array2::<i8>::zeros((block.n_sites(), block.n_full()));
        Zip::from(rhap.rows_mut())
            .and(block.haplotypes.rows())
            .for_each(|mut row, ref_row| {
                let ref_row = BitSlice::<u8, Lsb0>::from_slice(ref_row.as_slice().unwrap());
                Zip::from(&mut row)
                    .and(&block.index_map)
                    .for_each(|geno, &index| {
                        *geno = ref_row[index as usize] as i8;
                    })
            });
        Self { rhap }
    }
}

pub fn m3vcf_block_scan(
    block: &m3vcf::Block,
    filter: &[bool],
    is_first_block: bool,
    afreqs: &mut Vec<f64>,
) -> Option<Block> {
    let n_skip = if is_first_block { 0 } else { 1 };

    let n_filtered_sites = filter.iter().skip(n_skip).filter(|b| **b).count();

    if n_filtered_sites == 0 {
        return None;
    }

    let mut filtered_haps = Array2::<u8>::zeros((n_filtered_sites, block.rhap.ncols()));

    let ref_haps_iter = block
        .rhap
        .rows()
        .into_iter()
        .skip(n_skip)
        .zip(block.afreq.iter().skip(n_skip))
        .zip(filter.iter().skip(n_skip))
        .filter_map(|((row, afreq), &b)| if b { Some((row, afreq)) } else { None });

    for (mut row, (ref_row, &afreq)) in filtered_haps.rows_mut().into_iter().zip(ref_haps_iter) {
        afreqs.push(afreq.into());
        row.assign(&ref_row);
    }

    //let mut weights = Array1::<f64>::zeros(block.nuniq);
    //for &i in &block.indmap {
    //weights[i as usize] += 1.;
    //}

    let index_map = block.indmap.map(|&i| i as usize);

    Some(Block {
        index_map,
        haplotypes: filtered_haps,
        //inv_weights: weights.map(|v| 1. / v),
        //weights,
        n_unique: block.nuniq,
    })
}
