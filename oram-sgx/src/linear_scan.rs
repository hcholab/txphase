use crate::{Block, BlockSize, ORAMBackend, ORAMBackendCreator};
use mc_oblivious_traits::{subtle::ConstantTimeEq, A64Bytes, CMov};

pub struct LinearScanningORAMCreator;

impl ORAMBackendCreator for LinearScanningORAMCreator {
    type Backend = LinearScanningORAM;
    fn new_empty(&self, n_blocks: usize) -> Self::Backend {
        LinearScanningORAM {
            data: vec![Default::default(); n_blocks as usize],
        }
    }

    fn new_init<'a>(
        &self,
        blocks_iter: impl Iterator<Item = std::borrow::Cow<'a, Block>> + ExactSizeIterator,
    ) -> Self::Backend {
        let data = blocks_iter.map(|v| v.into_owned()).collect::<Vec<_>>();
        LinearScanningORAM { data }
    }
}

pub struct LinearScanningORAM {
    data: Vec<A64Bytes<BlockSize>>,
}

impl ORAMBackend for LinearScanningORAM {
    fn read_block(&mut self, block_index: usize) -> Block {
        let mut temp: A64Bytes<BlockSize> = Default::default();
        for idx in 0..self.data.len() {
            temp.cmov((idx as u64).ct_eq(&(block_index as u64)), &self.data[idx]);
        }
        temp
    }

    fn write_block(&mut self, block: &Block, block_index: usize) {
        for idx in 0..self.data.len() {
            self.data[idx].cmov((idx as u64).ct_eq(&(block_index as u64)), block);
        }
    }
}
