mod block;
pub use block::*;

use ndarray::Array1;

////const MAX_UNIQUE: usize = 100000;
//const MAX_UNIQUE: usize = 800;
const MIN_UNIQUE: usize = 300;

pub fn m3vcf_scan(
    meta: &m3vcf::RefPanelMeta,
    m3vcf_blocks: &[m3vcf::Block],
    sites_bitmask: &[bool],
    min_unique: Option<usize>,
    max_unique: Option<usize>,
) -> (RefPanel, Vec<f64>) {
    assert_eq!(sites_bitmask.len(), meta.n_markers);
    let mut afreqs = Vec::new();
    let mut pos = 0;

    let min_unique = min_unique.unwrap_or(meta.n_haps / 200);
    let max_unique = max_unique.unwrap_or(meta.n_haps / 2);

    let mut blocks = Vec::new();
    //let mut transposed_blocks = Vec::new();

    let mut cur_block: Option<Block> = None;

    for (i, m3vcf_block) in m3vcf_blocks.iter().enumerate() {
        let nvar = if i == 0 {
            m3vcf_block.nvar
        } else {
            m3vcf_block.nvar - 1
        };

        let start_pos = if i == 0 { pos } else { pos - 1 };
        let sites_bitmask_block = if i == m3vcf_blocks.len() - 1 {
            &sites_bitmask[start_pos..]
        } else {
            &sites_bitmask[start_pos..pos + nvar]
        };

        pos += nvar;

        if let Some(block) = m3vcf_block_scan(m3vcf_block, sites_bitmask_block, i == 0, &mut afreqs)
        {
            if let Some(cur_block_) = cur_block.take() {
                if cur_block_.n_unique() < MIN_UNIQUE || cur_block_.n_unique() < min_unique {
                    let merge_block = merge_blocks(&cur_block_, &block);
                    cur_block = Some(merge_block);
                } else if cur_block_.n_unique() > max_unique {
                    blocks.extend(break_block(&cur_block_, max_unique).into_iter());
                    cur_block = Some(block);
                } else {
                    let cur_block_ = make_unique_hap_block(&cur_block_);
                    //transposed_blocks.push(cur_block_.transpose());
                    blocks.push(cur_block_);

                    //blocks.push(make_unique_hap_block(&cur_block_));
                    cur_block = Some(block);
                }
            } else {
                cur_block = Some(block);
            }
        }
    }

    if let Some(cur_block) = cur_block.take() {
        if cur_block.n_unique() > max_unique {
            blocks.extend(break_block(&cur_block, max_unique).into_iter());
        } else {
            let new_block = make_unique_hap_block(&cur_block);
            blocks.push(new_block);
        }
        //transposed_blocks.push(new_block.transpose());
    }

    //println!("n_blocks: {}", blocks.len());
    //for block in &blocks {
    //println!(
    //"n_unique: {},\t n_sites: {}",
    //block.n_unique(),
    //block.n_sites()
    //);
    //}

    //for i in 1..blocks.len() {
    //let (block1, block2) = blocks.split_at_mut(i);
    //let block1 = block1.last_mut().unwrap();
    //let block2 = block2.first_mut().unwrap();
    //let index_map_1 = &block1.index_map;
    //let index_map_2 = &block2.index_map;

    //if let Some(start_overlap_hap) = block2.start_overlap_hap.as_ref() {
    //let start_overlap_hap_bit =
    //BitSlice::<Lsb0, u8>::from_slice(start_overlap_hap.as_slice().unwrap()).unwrap();
    //let mut end_overlap_hap = Array1::<u8>::zeros(block1.haplotypes.ncols());
    //let end_overlap_hap_bit =
    //BitSlice::<Lsb0, u8>::from_slice_mut(end_overlap_hap.as_slice_mut().unwrap())
    //.unwrap();
    //for (&i1, &i2) in index_map_1.iter().zip(index_map_2.iter()) {
    //end_overlap_hap_bit.set(i1, start_overlap_hap_bit[i2]);
    //}
    //} else {
    //rebuild_blocks(block1, block2, meta.n_haps);
    //panic!()
    //}
    //}

    let block_map = gen_block_map(&blocks);

    let ref_panel = RefPanel {
        blocks,
        //transposed_blocks,
        block_map,
        n_haps: meta.n_haps,
        n_sites: afreqs.len(),
    };

    (ref_panel, afreqs)
}

pub struct RefPanel {
    pub blocks: Vec<Block>,
    //pub transposed_blocks: Vec<TransposedBlock>,
    pub block_map: Vec<usize>,
    pub n_haps: usize,
    pub n_sites: usize,
}

impl RefPanel {
    pub fn slice<'a>(&'a self, start: usize, end: usize) -> RefPanelSlice<'a> {
        let start_block_id = match self.block_map.binary_search(&start) {
            Ok(i) => i + 1,
            Err(i) => i,
        };
        let end_block_id = match self.block_map.binary_search(&end) {
            Ok(i) => i,
            Err(i) => i,
        };

        let start_offset = if start_block_id == 0 {
            start
        } else {
            start - self.block_map[start_block_id - 1]
        };

        let end_offset = if end_block_id == 0 {
            end
        } else {
            end - self.block_map[end_block_id - 1]
        };

        let mut block_slices = Vec::with_capacity(end_block_id - start_block_id + 1);
        //let mut transposed_block_slices = Vec::with_capacity(end_block_id - start_block_id + 1);

        let start_block_slice =
            self.blocks[start_block_id].slice(start_offset, self.blocks[start_block_id].n_sites());
        //let start_transposed_block_slice = self.transposed_blocks[start_block_id]
        //.slice(start_offset, self.blocks[start_block_id].n_sites());

        let end_block_slice = self.blocks[end_block_id].slice(0, end_offset);
        //let end_transposed_block_slice = self.transposed_blocks[end_block_id].slice(0, end_offset);

        block_slices.push(start_block_slice);
        //transposed_block_slices.push(start_transposed_block_slice);

        for i in (start_block_id + 1)..end_block_id {
            block_slices.push(self.blocks[i].as_slice());
            //transposed_block_slices.push(self.transposed_blocks[i].as_slice());
        }

        block_slices.push(end_block_slice);
        //transposed_block_slices.push(end_transposed_block_slice);

        let n_sites = block_slices.iter().map(|b| b.n_sites()).sum();

        RefPanelSlice {
            blocks: block_slices,
            //transposed_blocks: transposed_block_slices,
            n_haps: self.n_haps,
            n_sites,
        }
    }

    pub fn iter<'a>(&'a self) -> RowIterator<'a, Box<dyn Iterator<Item = BlockSlice<'a>> + 'a>> {
        RowIterator::from_blocks_iter(Box::new(self.blocks.iter().map(|b| b.as_slice())))
    }
}

pub struct RefPanelSlice<'a> {
    pub blocks: Vec<BlockSlice<'a>>,
    //pub transposed_blocks: Vec<TransposedBlockSlice<'a>>,
    pub n_haps: usize,
    pub n_sites: usize,
}

//use crate::types::Genotype;
//use ndarray::{s, Array2};

impl<'a> RefPanelSlice<'a> {
    pub fn iter(&self) -> RowIterator<'a, Box<dyn Iterator<Item = BlockSlice<'a>> + 'a>> {
        RowIterator::from_blocks_iter(Box::new(self.blocks.clone().into_iter()))
    }

    //pub fn filter(&self, bitmask: &[bool]) -> Array2<Genotype> {
    //let n_haps = bitmask.iter().filter(|b| **b).count();
    //let mut ref_panel = Array2::<Genotype>::zeros((self.n_sites, n_haps));
    //let mut pos = 0;
    //for block in self.transposed_blocks.iter() {
    //block.filter(
    //bitmask,
    //ref_panel.slice_mut(s![pos..pos + block.n_sites(), ..]),
    //);
    //pos += block.n_sites();
    //}
    //ref_panel
    //}
}

fn gen_block_map(blocks: &[Block]) -> Vec<usize> {
    let mut map = Vec::with_capacity(blocks.len());
    let mut s = 0;
    for block in blocks.iter() {
        s += block.n_sites();
        map.push(s);
    }
    map
}

pub struct RowIterator<'a, B>
where
    B: Iterator<Item = BlockSlice<'a>>,
{
    blocks_iterator: B,
    cur_i: usize,
    cur_block: ExpandedBlock,
}

impl<'a, B> RowIterator<'a, B>
where
    B: Iterator<Item = BlockSlice<'a>>,
{
    pub fn from_blocks_iter(mut blocks_iterator: B) -> Self {
        let cur_block = ExpandedBlock::from_block_slice(blocks_iterator.next().unwrap());
        Self {
            blocks_iterator,
            cur_i: 0,
            cur_block,
        }
    }
}

impl<'a, B> Iterator for RowIterator<'a, B>
where
    B: Iterator<Item = BlockSlice<'a>>,
{
    type Item = Array1<i8>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.cur_i == self.cur_block.rhap.nrows() {
                if let Some(new_block) = self.blocks_iterator.next() {
                    self.cur_block = ExpandedBlock::from_block_slice(new_block);
                    self.cur_i = 0; // skip the first overlapping position
                } else {
                    return None;
                }
            }
            let i = self.cur_i;
            self.cur_i += 1;
            return Some(self.cur_block.rhap.row(i).to_owned());
        }
    }
}
