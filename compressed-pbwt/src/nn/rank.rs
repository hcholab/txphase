#[derive(Debug, PartialEq, PartialOrd, Eq, Ord, Clone, Copy, Default)]
pub struct NNRank((usize, usize, usize));

impl NNRank {
    pub fn new(div: usize, dist: usize, is_below: bool, hap_id: usize) -> Self {
        Self((div, dist << 1 | (is_below as usize), hap_id))
    }

    pub fn get_hap_id(&self) -> usize {
        self.0 .2
    }
}
