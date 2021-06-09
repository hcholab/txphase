use crate::align::A64Bytes;
use crate::backend::*;
use crate::utils::tp_u32_div;
use std::borrow::Cow;
use timing_shield::TpU32;

pub struct SmallORAM<C: BaseORAMBackendCreator<64>> {
    n_values: u32,
    value_size: usize,
    values_per_block: u32,
    pub(crate) backend: C::ORAM,
}

impl<C: BaseORAMBackendCreator<64>> SmallORAM<C> {
    pub fn new(n_values: u32, value_size: usize, oram_backend_creator: C) -> Self {
        assert!(value_size <= 64);
        let values_per_block = (64 / value_size) as u32;
        let n_blocks = (n_values + values_per_block - 1) / values_per_block;
        let backend = oram_backend_creator.new(n_blocks);
        Self {
            n_values,
            value_size,
            values_per_block,
            backend,
        }
    }
}

impl<C: ReadOnlyORAMBackendCreator<64>> SmallORAM<C>
where
    C::ORAM: ReadOnlyORAMBackend<64>,
{
    pub fn from_iter<'a>(
        mut values_iter: impl Iterator<Item = &'a [u8]> + ExactSizeIterator,
        value_size: usize,
        oram_backend_creator: C,
    ) -> Self {
        assert!(value_size <= 64);
        use std::convert::TryInto;
        let n_values: u32 = values_iter.len().try_into().unwrap();
        let values_per_block = (64 / value_size) as u32;
        let n_blocks = (n_values + values_per_block - 1) / values_per_block;
        let iter = (0..n_blocks).map(move |_| {
            let mut new_block = A64Bytes::<64>::default();
            for block_slice in new_block.as_mut_slice().chunks_mut(value_size) {
                if block_slice.len() != value_size {
                    break;
                }
                if let Some(value) = values_iter.next() {
                    assert_eq!(value.len(), value_size);
                    block_slice.copy_from_slice(value);
                } else {
                    break;
                }
            }
            Cow::Owned(new_block)
        });

        let backend = oram_backend_creator.from_iter(iter);
        Self {
            n_values,
            value_size,
            values_per_block,
            backend,
        }
    }
    // Assuming reading from a cache line is safe
    pub fn read(&mut self, index: TpU32) -> Vec<u8> {
        let (block_index, index_in_block) =
            tp_u32_div(index, self.values_per_block as u32, self.n_values as u32);
        let block = self.backend.read_block(block_index);
        let mut result = vec![0u8; self.value_size];
        let exposed_i = index_in_block.expose() as usize;
        result.copy_from_slice(
            &(block.as_slice())[(exposed_i * self.value_size)..((exposed_i + 1) * self.value_size)],
        );
        result
    }
}

impl<C: WriteOnlyORAMBackendCreator<64>> SmallORAM<C>
where
    C::ORAM: WriteOnlyORAMBackend<64>,
{
    // Assuming writing to a cache line is safe
    pub fn write(&mut self, index: TpU32, value: &[u8]) {
        assert_eq!(value.len(), self.value_size);
        self.modify_block_with(index, |block_slice| block_slice.copy_from_slice(&value[..]))
    }

    pub fn modify_block_with(&mut self, index: TpU32, func: impl Fn(&mut [u8])) {
        let (block_index, index_in_block) =
            tp_u32_div(index, self.values_per_block as u32, self.n_values as u32);
        let exposed_i = index_in_block.expose() as usize;
        let value_size = self.value_size;
        self.backend.fast_modify_block_with(block_index, |block| {
            func(
                &mut (block.as_mut_slice())
                    [(exposed_i * value_size)..((exposed_i + 1) * value_size)],
            )
        });
    }

    pub fn into_iter(self) -> Box<dyn Iterator<Item = Vec<u8>>> {
        let (backend, n_values, value_size, values_per_block) = (
            self.backend,
            self.n_values,
            self.value_size,
            self.values_per_block,
        );
        let n_blocks = (n_values + values_per_block - 1) / values_per_block;
        let mut remaining_values = n_values;
        Box::new(
            backend
                .into_iter()
                .take(n_blocks as usize)
                .map(move |block| {
                    let take_n_values = remaining_values.min(values_per_block);
                    remaining_values -= take_n_values;
                    block
                        .as_slice()
                        .chunks(value_size)
                        .take(take_n_values as usize)
                        .map(|chunk| {
                            let mut value = vec![0u8; value_size];
                            value.copy_from_slice(chunk);
                            value
                        })
                        .collect::<Vec<_>>()
                })
                .flatten(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{prelude::SliceRandom, RngCore};

    fn test_template<C: RWORAMBackendCreator<64>>(oram_backend_creator: C)
    where
        <C as BaseORAMBackendCreator<64>>::ORAM: ReadOnlyORAMBackend<64> + WriteOnlyORAMBackend<64>,
    {
        let value_size = 7;
        let n_values = 1000;
        let n_access = 100;
        let mut values = {
            let mut counter = 0u8;
            (0..n_values)
                .map(|_| {
                    let mut value = vec![0u8; value_size];
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

        let mut oram = SmallORAM::<_>::from_iter(
            values.iter().map(|v| v.as_slice()),
            value_size,
            oram_backend_creator,
        );

        for &index in &access_indices {
            let result = oram.read(TpU32::protect(index as u32));
            assert_eq!(result, values[index]);
        }

        let mut rng = rand::thread_rng();
        for &index in &access_indices {
            let mut mask = vec![0u8; value_size];
            rng.fill_bytes(&mut mask);

            let func = |block: &mut [u8]| {
                for (byte, mask_byte) in block.iter_mut().zip(mask.iter()) {
                    *byte &= mask_byte;
                }
            };
            oram.modify_block_with(TpU32::protect(index as u32), func);
            let result = oram.read(TpU32::protect(index as u32));
            func(&mut values[index]);
            assert_eq!(result, values[index]);
        }

        let oram_values = oram.into_iter().collect::<Vec<_>>();
        for &index in &access_indices {
            assert_eq!(oram_values[index], values[index]);
        }
    }

    //#[test]
    //fn leaky() {
    //test_template(LeakyORAMCreator);
    //}

    #[test]
    fn linear_scan() {
        test_template(LinearScanningORAMCreator);
    }

    //#[test]
    //fn path_oram() {
    //test_template(PathORAMCreator::with_stash_size(11));
    //}
}
