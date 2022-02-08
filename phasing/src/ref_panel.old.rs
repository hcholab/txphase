use crate::block::RowIterator;
use m3vcf::{Block, BlockView, RefPanelMeta};

pub struct RefPanel {
    pub blocks: Vec<Block>,
    pub n_haps: usize,
    pub sites_bitmask: Vec<bool>,
    sites_pos: Vec<usize>,
    blocks_map: Vec<usize>,
}

impl RefPanel {
    pub fn new(
        meta: RefPanelMeta,
        blocks: Vec<Block>,
        sites_bitmask: Vec<bool>,
    ) -> Self {
        let blocks_map = gen_block_map(&blocks);
        let mut sites_pos = sites_bitmask
            .iter()
            .enumerate()
            .filter(|(_, b)| **b)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        sites_pos.push(sites_bitmask.len());

        Self {
            blocks,
            n_haps: meta.n_haps,
            sites_bitmask,
            sites_pos,
            blocks_map,
        }
    }

    pub fn as_slice<'a>(&'a self) -> RefPanelSlice<'a> {
        RefPanelSlice{
            blocks: self.blocks.iter().map(|b| b.view()).collect(),
            n_haps: self.n_haps,
            sites_bitmask: self.sites_bitmask.as_slice(),
        }
    }

    pub fn slice<'a>(&'a self, start: usize, end: usize) -> RefPanelSlice<'a> {
        let start = self.sites_pos[start];
        let end = self.sites_pos[end];
        let blocks = sub_blocks(&self.blocks, &self.blocks_map, start, end);
        RefPanelSlice {
            blocks,
            n_haps: self.n_haps,
            sites_bitmask: &self.sites_bitmask[start..end],
        }
    }

    pub fn iter<'a>(
        &'a self,
    ) -> RowIterator<
        'a,
        'a,
        Box<dyn Iterator<Item = BlockView<'a>> + 'a>,
        Box<dyn Iterator<Item = &'a bool> + 'a>,
    > {
        RowIterator::from_blocks_iter(
            Box::new(self.blocks.iter().map(|b| b.view())),
            Box::new(self.sites_bitmask.iter()),
        )
    }
}

pub struct RefPanelSlice<'a> {
    pub blocks: Vec<BlockView<'a>>,
    pub n_haps: usize,
    pub sites_bitmask: &'a [bool],
}

impl<'a> RefPanelSlice<'a> {
    pub fn iter(
        &self,
    ) -> RowIterator<
        'a,
        'a,
        Box<dyn Iterator<Item = BlockView<'a>> + 'a>,
        Box<dyn Iterator<Item = &'a bool> + 'a>,
    > {
        RowIterator::from_blocks_iter(
            Box::new(self.blocks.clone().into_iter()),
            Box::new(self.sites_bitmask.iter()),
        )
    }
}

fn gen_block_map(blocks: &[Block]) -> Vec<usize> {
    let mut map = vec![0; blocks.len()];
    let mut s = 1;
    for (block, v) in blocks.iter().zip(map.iter_mut().skip(1)) {
        s += block.nvar - 1;
        *v = s;
    }
    map
}

fn sub_blocks<'a>(
    blocks: &'a [Block],
    block_map: &[usize],
    from: usize,
    to: usize,
) -> Vec<BlockView<'a>> {
    assert!(to > from);
    assert!(to != 0);
    let from_block_id = match block_map.binary_search(&from) {
        Ok(i) => i,
        Err(i) => i - 1,
    };
    let to_block_id = match block_map.binary_search(&to) {
        Ok(i) => {
            if i == 0 {
                i
            } else {
                i - 1
            }
        }
        Err(i) => i - 1,
    };

    let from_offset = if from_block_id == 0 {
        from
    } else {
        from - block_map[from_block_id] + 1
    };

    let to_offset = if to_block_id == 0 {
        to
    } else {
        to - block_map[to_block_id] + 1
    };

    let from_block_view = blocks[from_block_id].subview(from_offset, blocks[from_block_id].nvar);
    let to_block_view = blocks[to_block_id].subview(0, to_offset);
    let mut blockviews = Vec::with_capacity(to_block_id - from_block_id + 1);
    blockviews.push(from_block_view);
    for i in (from_block_id + 1)..to_block_id {
        blockviews.push(blocks[i].view())
    }
    blockviews.push(to_block_view);
    blockviews
}

