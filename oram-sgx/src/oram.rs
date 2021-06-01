use crate::{Block, BlockSize, ORAMBackend, ORAMBackendCreator};
use mc_oblivious_traits::typenum::Unsigned;
use std::borrow::Cow;

pub struct ORAM<OBC: ORAMBackendCreator, const VALUE_SIZE: usize> {
    values_per_block: usize,
    backend: OBC::Backend,
}

impl<OBC: ORAMBackendCreator, const VALUE_SIZE: usize> ORAM<OBC, VALUE_SIZE> {
    pub fn new_empty(n_values: usize, oram_backend_creator: OBC) -> Self {
        assert!(VALUE_SIZE <= 64);
        let values_per_block = BlockSize::USIZE / VALUE_SIZE;
        let n_blocks = (n_values + values_per_block - 1) / values_per_block;
        let backend = oram_backend_creator.new_empty(n_blocks);
        Self {
            values_per_block,
            backend,
        }
    }

    pub fn new_init<'a>(
        mut values_iter: impl Iterator<Item = Cow<'a, [u8; VALUE_SIZE]>> + ExactSizeIterator,
        oram_backend_creator: OBC,
    ) -> Self {
        let values_per_block = BlockSize::USIZE / VALUE_SIZE;
        let n_blocks = (values_iter.len() + values_per_block - 1) / values_per_block;
        let iter = (0..n_blocks).map(move |_| {
            let mut inblock_iter: Box<dyn Iterator<Item = u8>> = Box::new(std::iter::empty());
            for _ in 0..values_per_block {
                if let Some(value) = values_iter.next() {
                    let value = value.to_vec();
                    inblock_iter = Box::new(inblock_iter.chain(value.into_iter()));
                } else {
                    break;
                }
            }
            inblock_iter = Box::new(inblock_iter.chain(std::iter::repeat(0u8)));
            use std::iter::FromIterator;
            let block = Block::from_iter(inblock_iter);
            std::borrow::Cow::<Block>::Owned(block)
        });

        let backend = oram_backend_creator.new_init(iter);
        Self {
            values_per_block,
            backend,
        }
    }

    // Assuming reading from a cache line is safe
    pub fn read(&mut self, index: usize) -> [u8; VALUE_SIZE] {
        let block_index = index / self.values_per_block;
        let index_in_block = index % self.values_per_block;
        let block = self.backend.read_block(block_index);
        let mut result = [0u8; VALUE_SIZE];
        result.copy_from_slice(
            &block[(index_in_block * VALUE_SIZE)..((index_in_block + 1) * VALUE_SIZE)],
        );
        result
    }
}
