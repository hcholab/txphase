mod block;
pub use block::*;

use crate::Genotype;
use ndarray::{s, Array1, Array2, ArrayView2};

pub fn m3vcf_scan(
    meta: &m3vcf::RefPanelMeta,
    m3vcf_blocks: &[m3vcf::Block],
    sites_bitmask: &[bool],
) -> (RefPanel, Vec<f64>) {
    assert_eq!(sites_bitmask.len(), meta.n_markers);
    let mut afreqs = Vec::new();
    let mut pos = 0;

    let mut blocks = Vec::new();
    let mut transposed_blocks = Vec::new();


    for (i, m3vcf_block) in m3vcf_blocks.iter().enumerate() {
        let nvar = if i == 0 {
            m3vcf_block.nvar
        } else {
            m3vcf_block.nvar - 1
        };

        let sites_bitmask_block = if i == m3vcf_blocks.len() - 1 {
            &sites_bitmask[pos..]
        } else {
            &sites_bitmask[pos..pos + nvar]
        };

        pos += nvar;

        if let Some(block) = m3vcf_block_scan(m3vcf_block, sites_bitmask_block, i == 0, &mut afreqs)
        {
            transposed_blocks.push(block.transpose());
            blocks.push(block);
        }
    }

    //TODO fix this
    //let full_ref = Array2::<i8>::zeros((0, 0));

    let block_map = gen_block_map(&blocks);

    let ref_panel = RefPanel {
        blocks,
        transposed_blocks,
        //full_ref,
        block_map,
        n_haps: meta.n_haps,
        n_sites: afreqs.len(),
    };

    //let mut full_ref = Array2::<i8>::zeros((afreqs.len(), meta.n_haps));
    //let mut col_iter = full_ref.rows_mut().into_iter();
    //let mut cur_col = col_iter.next();
    //let cur_block_iter = ref_panel.blocks.iter().map(|b| ExpandedBlock::from_block_slice(b.as_slice()));
    //for cur_block in cur_block_iter {
        //for block_col in cur_block.rhap.rows().into_iter() {
            //if let Some(mut cur_col_) = cur_col {
                //cur_col_.assign(&block_col);
                //cur_col = col_iter.next();
            //}
        //}
    //}

    //ref_panel.full_ref = full_ref;

    (ref_panel, afreqs)
}

pub struct RefPanel {
    pub blocks: Vec<Block>,
    //pub full_ref: Array2<i8>,
    pub transposed_blocks: Vec<TransposedBlock>,
    block_map: Vec<usize>,
    pub n_haps: usize,
    pub n_sites: usize,
}

impl RefPanel {
    pub fn slice<'a>(&'a self, start: usize, end: usize) -> RefPanelSlice<'a> {
        //let full_ref = self.full_ref.slice(s![start..end, ..]).to_owned();
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
        let mut transposed_block_slices = Vec::with_capacity(end_block_id - start_block_id + 1);

        let start_block_slice =
            self.blocks[start_block_id].slice(start_offset, self.blocks[start_block_id].n_sites());
        let start_transposed_block_slice = self.transposed_blocks[start_block_id]
            .slice(start_offset, self.blocks[start_block_id].n_sites());

        let end_block_slice = self.blocks[end_block_id].slice(0, end_offset);
        let end_transposed_block_slice = self.transposed_blocks[end_block_id].slice(0, end_offset);

        block_slices.push(start_block_slice);
        transposed_block_slices.push(start_transposed_block_slice);

        for i in (start_block_id + 1)..end_block_id {
            block_slices.push(self.blocks[i].as_slice());
            transposed_block_slices.push(self.transposed_blocks[i].as_slice());
        }

        block_slices.push(end_block_slice);
        transposed_block_slices.push(end_transposed_block_slice);

        let n_sites = block_slices.iter().map(|b| b.n_sites()).sum();

        RefPanelSlice {
            blocks: block_slices,
            //full_ref,
            transposed_blocks: transposed_block_slices,
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
    //pub full_ref: Array2<i8>,
    pub transposed_blocks: Vec<TransposedBlockSlice<'a>>,
    pub n_haps: usize,
    pub n_sites: usize,
}

impl<'a> RefPanelSlice<'a> {
    pub fn iter(&self) -> RowIterator<'a, Box<dyn Iterator<Item = BlockSlice<'a>> + 'a>> {
        RowIterator::from_blocks_iter(Box::new(self.blocks.clone().into_iter()))
    }
    //pub fn iter(&self) -> Box<dyn Iterator<Item = Array1<i8>> + '_> {
        //Box::new(self.full_ref.rows().into_iter().map(|r| r.to_owned()))
    //}

    pub fn filter(&self, bitmask: &[bool]) -> Array2<Genotype> {
        let n_haps = bitmask.iter().filter(|b| **b).count();
        let mut ref_panel = Array2::<Genotype>::zeros((self.n_sites, n_haps));
        let mut pos = 0;
        for block in self.transposed_blocks.iter() {
            block.filter(
                bitmask,
                ref_panel.slice_mut(s![pos..pos + block.n_sites(), ..]),
            );
            pos += block.n_sites();
        }
        ref_panel
    }
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
