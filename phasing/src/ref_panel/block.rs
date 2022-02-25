//use crate::oram::DynamicLSOram;
use crate::Genotype;
use bitvec::prelude::{BitSlice, Lsb0};
use ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, Zip};
use oram_sgx::align::A64Bytes;

pub struct Block {
    index_map: Array1<u16>,
    haplotypes: Array2<u8>,
}

impl Block {
    pub fn n_sites(&self) -> usize {
        self.haplotypes.nrows()
    }

    pub fn n_unique(&self) -> usize {
        self.index_map.len()
    }

    pub fn as_slice<'a>(&'a self) -> BlockSlice<'a> {
        BlockSlice {
            index_map: self.index_map.view(),
            haplotypes: self.haplotypes.view(),
        }
    }

    pub fn slice<'a>(&'a self, start: usize, end: usize) -> BlockSlice<'a> {
        BlockSlice {
            index_map: self.index_map.view(),
            haplotypes: self.haplotypes.slice(s![start..end, ..]),
        }
    }

    pub fn transpose(&self) -> TransposedBlock {
        let ncols = (self.n_sites() + 7) / 8;
        let mut transposed_haplotypes =
            unsafe { Array2::<u8>::uninit((self.n_unique(), ncols)).assume_init() };
        Zip::indexed(self.haplotypes.rows()).for_each(|i, s| {
            let s = BitSlice::<Lsb0, u8>::from_slice(s.as_slice().unwrap()).unwrap();
            for (b, mut hap) in s.iter().zip(transposed_haplotypes.rows_mut().into_iter()) {
                let hap =
                    BitSlice::<Lsb0, u8>::from_slice_mut(hap.as_slice_mut().unwrap()).unwrap();
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

#[derive(Clone)]
pub struct BlockSlice<'a> {
    index_map: ArrayView1<'a, u16>,
    haplotypes: ArrayView2<'a, u8>,
}

impl<'a> BlockSlice<'a> {
    pub fn n_sites(&self) -> usize {
        self.haplotypes.nrows()
    }

    pub fn n_unique(&self) -> usize {
        self.index_map.len()
    }
}

pub struct TransposedBlock {
    index_map: Array1<u16>,
    haplotypes: Array2<u8>,
    n_sites: usize,
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
    index_map: ArrayView1<'a, u16>,
    haplotypes: ArrayView2<'a, u8>,
    n_sites: usize,
    range: Option<(usize, usize)>,
}

impl<'a> TransposedBlockSlice<'a> {
    pub fn filter(&self, bitmask: &[bool], mut output: ArrayViewMut2<Genotype>) {
        assert_eq!(output.nrows(), self.n_sites);
        let mut hap_count = 0;
        for (i, b) in bitmask.iter().enumerate() {
            if *b {
                let hap_i = self.index_map[i] as usize;
                let tmp = self.haplotypes.row(hap_i);
                let hap = BitSlice::<Lsb0, u8>::from_slice(tmp.as_slice().unwrap()).unwrap();
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
        let mut rhap = Array2::<i8>::zeros((block.n_sites(), block.n_unique()));
        Zip::from(rhap.rows_mut())
            .and(block.haplotypes.rows())
            .for_each(|mut row, ref_row| {
                let ref_row =
                    BitSlice::<Lsb0, u8>::from_slice(ref_row.as_slice().unwrap()).unwrap();
                Zip::from(&mut row)
                    .and(&block.index_map)
                    .for_each(|geno, &index| {
                        *geno = ref_row[index as usize] as i8;
                    })
            });
        Self { rhap }
    }
}

//pub fn block_to_aligned_transposed(block: &Block, n_haps: usize) -> DynamicLSOram<i8> {
//let mut transposed = Array2::<i8>::zeros((n_haps, block.n_sites()));
//Zip::indexed(block.haplotypes.rows()).for_each(|i, row| {
//let row = BitSlice::<Lsb0, u8>::from_slice(row.as_slice().unwrap()).unwrap();
//for (mut t, &index) in transposed
//.rows_mut()
//.into_iter()
//.zip(block.index_map.iter())
//{
//t[i] = row[index as usize] as i8;
//}
//});
//let transposed = DynamicLSOram::from_array(transposed.view());
//transposed
//}

pub fn block_to_aligned_transposed<'a, const N: usize>(
    block: &BlockSlice,
    nhap: usize,
) -> Vec<A64Bytes<N>> {
    let mut transposed = vec![A64Bytes::<N>::default(); nhap];
    Zip::indexed(block.haplotypes.rows()).for_each(|i, row| {
        let row = BitSlice::<Lsb0, u8>::from_slice(row.as_slice().unwrap()).unwrap();
        for (t, &index) in transposed.iter_mut().zip(block.index_map.iter()) {
            assert!((index as usize) < row.len());
            t.as_mut_slice()[i] = row[index as usize] as u8;
        }
    });
    transposed
}

pub fn m3vcf_block_scan(
    block: &m3vcf::Block,
    filter: &[bool],
    is_first_block: bool,
    afreqs: &mut Vec<f64>,
) -> Option<Block> {
    let n_filtered_sites = filter.iter().filter(|b| **b).count();

    if n_filtered_sites == 0 {
        return None;
    }

    let n_skip = if is_first_block { 0 } else { 1 };

    let mut filtered_haps = Array2::<u8>::zeros((n_filtered_sites, block.rhap.ncols()));

    let ref_haps_iter = block
        .rhap
        .rows()
        .into_iter()
        .skip(n_skip)
        .zip(block.afreq.iter().skip(n_skip))
        .zip(filter.iter())
        .filter_map(|((row, afreq), &b)| if b { Some((row, afreq)) } else { None });

    for (mut row, (ref_row, &afreq)) in filtered_haps.rows_mut().into_iter().zip(ref_haps_iter) {
        afreqs.push(afreq.into());
        row.assign(&ref_row);
    }

    //// debug
    //for (pos, &afreq) in filtered_haps
    //.rows()
    //.into_iter()
    //.zip(afreqs[afreqs.len() - n_filtered_sites..].iter())
    //{
    //let pos = BitSlice::<Lsb0, u8>::from_slice(pos.as_slice().unwrap()).unwrap();
    //let mut cref = 0;
    //let mut calt = 0;

    //for &j in &block.indmap {
    //if pos[j as usize] {
    //calt += 1;
    //} else {
    //cref += 1;
    //}
    //}
    //assert_eq!((afreq * (cref + calt) as f64).round() as usize, calt);
    //}

    Some(Block {
        index_map: block.indmap.clone(),
        haplotypes: filtered_haps,
    })
}
