use crate::{Block, ORAMBackend, ORAMBackendCreator};

#[derive(Clone)]
pub struct LeakyORAMCreator;

impl ORAMBackendCreator for LeakyORAMCreator {
    type Backend = LeakyORAM;
    fn new_empty(&self, n_blocks: usize) -> Self::Backend {
        LeakyORAM {
            data: vec![Default::default(); n_blocks],
        }
    }
    fn new_init<'a>(
        &self,
        blocks_iter: impl Iterator<Item = std::borrow::Cow<'a, Block>> + ExactSizeIterator,
    ) -> Self::Backend {
        LeakyORAM {
            data: blocks_iter.map(|v| v.into_owned()).collect(),
        }
    }
}

pub struct LeakyORAM {
    data: Vec<Block>,
}

impl ORAMBackend for LeakyORAM {
    fn read_block(&mut self, block_index: usize) -> Block {
        self.data[block_index].clone()
    }

    fn write_block(&mut self, block: &Block, block_index: usize) {
        self.data[block_index] = block.clone();
    }

    fn modify_block_with(&mut self, block_index: usize, func: impl Fn(&mut Block)) {
        func(&mut self.data[block_index])
    }
}
