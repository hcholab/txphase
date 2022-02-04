use crate::oram::DynamicLSOram;
use bitvec::prelude::{BitSlice, Lsb0};
use m3vcf::{Block, BlockView};
use ndarray::{Array1, Array2, Zip};
use oram_sgx::align::A64Bytes;

pub struct RowIterator<'a, 'b, B, F>
where
    B: Iterator<Item = BlockView<'a>>,
    F: Iterator<Item = &'b bool>,
{
    blocks_iterator: B,
    cur_i: usize,
    cur_block: ExpandedBlock,
    filter: F,
}

impl<'a, 'b, B, F> RowIterator<'a, 'b, B, F>
where
    B: Iterator<Item = BlockView<'a>>,
    F: Iterator<Item = &'b bool>,
{
    pub fn from_blocks_iter(mut blocks_iterator: B, filter: F) -> Self {
        let cur_block = ExpandedBlock::from_block_ref(blocks_iterator.next().unwrap());
        Self {
            blocks_iterator,
            cur_i: 0,
            cur_block,
            filter,
        }
    }
}

impl<'a, 'b, B, F> Iterator for RowIterator<'a, 'b, B, F>
where
    B: Iterator<Item = BlockView<'a>>,
    F: Iterator<Item = &'b bool>,
{
    type Item = Array1<i8>;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(b) = self.filter.next() {
                if self.cur_i == self.cur_block.rhap.nrows() {
                    if let Some(new_block) = self.blocks_iterator.next() {
                        self.cur_block = ExpandedBlock::from_block_ref(new_block);
                        self.cur_i = 1; // skip the first overlapping position
                    } else {
                        return None;
                    }
                }
                let i = self.cur_i;
                self.cur_i += 1;
                if *b {
                    return Some(self.cur_block.rhap.row(i).to_owned());
                }
            } else {
                return None;
            }
        }
    }
}

pub struct ExpandedBlock {
    pub rhap: Array2<i8>,
}

impl ExpandedBlock {
    pub fn from_block_ref<'a>(block: BlockView<'a>) -> Self {
        let n_ref_haps = block.indmap.len();
        let mut rhap = Array2::<i8>::zeros((block.nvar, n_ref_haps));
        Zip::from(rhap.rows_mut())
            .and(block.rhap.rows())
            .for_each(|mut row, ref_row| {
                let ref_row =
                    BitSlice::<Lsb0, u8>::from_slice(ref_row.as_slice().unwrap()).unwrap();
                Zip::from(&mut row)
                    .and(&block.indmap)
                    .for_each(|geno, &index| {
                        *geno = ref_row[index as usize] as i8;
                    })
            });
        Self { rhap }
    }
}

pub fn block_to_aligned_transposed<'a, const N: usize>(
    block: &BlockView,
    nhap: usize,
) -> Vec<A64Bytes<N>> {
    let mut transposed = vec![A64Bytes::<N>::default(); nhap];
    Zip::indexed(block.rhap.rows()).for_each(|i, row| {
        let row = BitSlice::<Lsb0, u8>::from_slice(row.as_slice().unwrap()).unwrap();
        for (t, &index) in transposed.iter_mut().zip(block.indmap.iter()) {
            assert!((index as usize) < row.len());
            t.as_mut_slice()[i] = row[index as usize] as u8;
        }
    });
    transposed
}

pub fn block_to_aligned_transposed_2(block: &Block, nhap: usize) -> DynamicLSOram<i8> {
    let mut transposed = Array2::<i8>::zeros((nhap, block.nvar));
    Zip::indexed(block.rhap.rows()).for_each(|i, row| {
        let row = BitSlice::<Lsb0, u8>::from_slice(row.as_slice().unwrap()).unwrap();
        for (mut t, &index) in transposed.rows_mut().into_iter().zip(block.indmap.iter()) {
            t[i] = row[index as usize] as i8;
        }
    });
    let transposed = DynamicLSOram::from_array(transposed.view());
    transposed
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::Array1;

    struct FakeBlock {
        pub indmap: Array1<u16>,
        pub nvar: usize,
        pub nuniq: usize,
        pub clustsize: Array1<u16>,
        pub rhap: Array2<u8>,
        pub rprob: Array1<f32>,
        pub afreq: Array1<f32>,
    }

    impl FakeBlock {
        fn random(nvar: usize, nuniq: usize, nhaps: usize) -> m3vcf::Block {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            let indmap = Array1::<u16>::from_shape_fn(nhaps, |_| rng.gen::<u16>() % nuniq as u16);
            let nuniq_bytes = (nuniq + 7) / 8;
            let rhap = Array2::<u8>::from_shape_fn((nvar, nuniq_bytes), |(_, _)| rng.gen::<u8>());

            let s = Self {
                indmap,
                nvar,
                nuniq,
                clustsize: Array1::zeros(0),
                rhap,
                rprob: Array1::zeros(0),
                afreq: Array1::zeros(0),
            };

            unsafe { std::mem::transmute(s) }
        }
    }

    #[test]
    fn from_block_ref_test() {
        let block = FakeBlock::random(100, 10, 20);
        let expanded = ExpandedBlock::from_block_ref(&block);
    }
}
