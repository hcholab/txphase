use crate::block::RowIterator;
use m3vcf::{Block, BlockView, RefPanelMeta};

pub struct RefPanel {
    pub meta: RefPanelMeta,
    pub blocks: Vec<Block>,
    pub blocks_map: Vec<usize>,
    pub sites_bitmask: Vec<bool>,
    pub sites_pos: Vec<usize>,
    pub cms: Vec<f32>,
    pub recomb_probs: Vec<f64>,
    pub rev_recomb_probs: Vec<f64>,
}

impl RefPanel {
    pub fn new(
        meta: RefPanelMeta,
        blocks: Vec<Block>,
        sites_bitmask: Vec<bool>,
        cms: Vec<f32>,
    ) -> Self {
        let blocks_map = gen_block_map(&blocks);
        let mut sites_pos = sites_bitmask
            .iter()
            .enumerate()
            .filter(|(_, b)| **b)
            .map(|(i, _)| i)
            .collect::<Vec<_>>();
        sites_pos.push(sites_bitmask.len());
        let cms = cms
            .into_iter()
            .zip(sites_bitmask.iter())
            .filter(|(_, b)| **b)
            .map(|(cm, _)| cm)
            .collect::<Vec<_>>();

        let n_eff = 15000;
        let (recomb_probs, rev_recomb_probs) = compute_all_recomb_probs(&cms, n_eff, meta.n_haps);

        Self {
            meta,
            blocks,
            blocks_map,
            sites_bitmask,
            sites_pos,
            cms,
            recomb_probs,
            rev_recomb_probs,
        }
    }

    pub fn view<'a>(&'a self) -> RefPanelView<'a> {
        RefPanelView {
            blocks: self.blocks.iter().map(|b| b.view()).collect(),
            meta: self.meta.clone(),
            sites_bitmask: self.sites_bitmask.as_slice(),
            cms: self.cms.as_slice(),
            recomb_probs: self.recomb_probs.as_slice(),
            rev_recomb_probs: self.rev_recomb_probs.as_slice(),
        }
    }

    pub fn sub_ref_panel<'a>(&'a self, start: usize, end: usize) -> RefPanelView<'a> {
        let recomb_probs = &self.recomb_probs[start..end];
        let rev_recomb_probs = &self.rev_recomb_probs[start..end];
        let cms = &self.cms[start..end];
        let start = self.sites_pos[start];
        let end = self.sites_pos[end];
        let blocks = sub_blocks(&self.blocks, &self.blocks_map, start, end);
        let meta = gen_ref_panel_meta(&blocks, self.meta.n_haps);
        RefPanelView {
            blocks,
            meta,
            sites_bitmask: &self.sites_bitmask[start..end],
            cms,
            recomb_probs,
            rev_recomb_probs,
        }
    }

    pub fn n_sites(&self) -> usize {
        self.recomb_probs.len()
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

pub struct RefPanelView<'a> {
    pub meta: RefPanelMeta,
    pub blocks: Vec<BlockView<'a>>,
    pub sites_bitmask: &'a [bool],
    pub cms: &'a [f32],
    pub recomb_probs: &'a [f64],
    pub rev_recomb_probs: &'a [f64],
}

impl<'a> RefPanelView<'a> {
    pub fn n_sites(&self) -> usize {
        self.recomb_probs.len()
    }

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

fn gen_ref_panel_meta(blocks: &[BlockView], n_haps: usize) -> RefPanelMeta {
    RefPanelMeta {
        n_haps,
        n_blocks: blocks.len(),
        n_markers: blocks.iter().map(|b| b.nvar).sum::<usize>() - blocks.len() + 1,
    }
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

fn compute_all_recomb_probs(cms: &[f32], n_eff: usize, n_haps: usize) -> (Vec<f64>, Vec<f64>) {
    let mut recomb_probs = Vec::with_capacity(cms.len());
    let mut rev_recomb_probs = Vec::with_capacity(cms.len());
    recomb_probs.push(0.);
    rev_recomb_probs.push(1.);
    for (prev, cur) in cms.iter().zip(cms.iter().skip(1)) {
        let r = compute_recomb_prob((cur - prev) as f64, n_eff, n_haps);
        recomb_probs.push(r);
        rev_recomb_probs.push(1. - r);
    }
    (recomb_probs, rev_recomb_probs)
}

fn compute_recomb_prob(dist_cm: f64, n_eff: usize, n_haps: usize) -> f64 {
    -1. * (-0.04 * n_eff as f64 * dist_cm / n_haps as f64).exp_m1()
}
