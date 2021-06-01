mod leaky;
mod linear_scan;
mod oram;
mod path_oram;

pub use leaky::*;
pub use linear_scan::*;
pub use oram::*;
pub use path_oram::*;

pub type BlockSize = U64;
pub type Block = A64Bytes<BlockSize>;

use mc_oblivious_traits::{typenum::U64, A64Bytes};

pub trait ORAMBackend {
    fn read_block(&mut self, block_index: usize) -> Block;
    fn write_block(&mut self, block: &Block, block_index: usize);
    fn modify_block_with(&mut self, block_index: usize, func: impl Fn(&mut Block));
}

pub trait ORAMBackendCreator {
    type Backend: ORAMBackend;
    fn new_empty(&self, n_blocks: usize) -> Self::Backend;
    fn new_init<'a>(
        &self,
        blocks_iter: impl Iterator<Item = std::borrow::Cow<'a, Block>> + ExactSizeIterator,
    ) -> Self::Backend;
}

#[macro_export]
macro_rules! oram_value {
    ($struct_name: ident) => {
        impl $struct_name {
            pub fn as_array(&self) -> &[u8; std::mem::size_of::<Self>()] {
                unsafe { std::mem::transmute(self) }
            }
        }

        impl From<[u8; std::mem::size_of::<Self>()]> for $struct_name {
            fn from(array: [u8; std::mem::size_of::<Self>()]) -> Self {
                unsafe { std::mem::transmute(array) }
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{prelude::SliceRandom, RngCore};
    use std::borrow::Cow;

    fn test_template(oram_backend_creator: impl ORAMBackendCreator) {
        const VALUE_SIZE: usize = 6;
        let n_values = 1000;
        let n_access = 100;
        let values = {
            let mut counter = 0u8;
            (0..n_values)
                .map(|_| {
                    let mut value = [0u8; VALUE_SIZE];
                    for byte in value.iter_mut() {
                        *byte = counter;
                        counter = counter.wrapping_add(1);
                    }
                    value
                })
                .collect::<Vec<_>>()
        };

        let access_indices = {
            let mut rng = rand::thread_rng();
            let range = (0..n_values).collect::<Vec<usize>>();
            range
                .choose_multiple(&mut rng, n_access)
                .cloned()
                .collect::<Vec<_>>()
        };

        let mut oram = ORAM::<_, VALUE_SIZE>::new_init(
            values.iter().map(|v| Cow::Borrowed(v)),
            oram_backend_creator,
        );

        for index in &access_indices {
            let result = oram.read(*index);
            assert_eq!(result, values[*index]);
        }

        let mut rng = rand::thread_rng();
        for index in &access_indices {
            let mut mask = [0u8; VALUE_SIZE];
            rng.fill_bytes(&mut mask);

            let func = |block: &mut [u8]| {
                for (byte, mask_byte) in block.iter_mut().zip(mask.iter()) {
                    *byte &= mask_byte;
                }
            };
            oram.modify_block_with(*index, func);
            let result = oram.read(*index);
            let mut ref_value = values[*index];
            func(&mut ref_value);
            assert_eq!(result, ref_value);
        }
    }

    #[test]
    fn leaky() {
        test_template(LeakyORAMCreator);
    }

    #[test]
    fn linear_scan() {
        test_template(LinearScanningORAMCreator);
    }

    #[test]
    fn path_oram() {
        test_template(PathORAMCreator::with_stash_size(11));
    }
}
