#[derive(Debug, PartialEq, PartialOrd, Eq, Ord, Clone, Copy, Default)]
pub struct InitRank((usize, usize));

impl InitRank {
    pub fn new(div: usize, dist: usize, is_below: bool, hap: bool) -> Self {
        Self((div, (dist << 2) | ((is_below as usize) << 1) | hap as usize))
    }

    pub fn get_div(&self) -> usize {
        self.0 .0
    }

    pub fn get_hap(&self) -> bool {
        (self.0 .1 & 1) == 1
    }

    pub fn set_hap(&mut self, hap: bool) {
        if hap {
            self.0 .1 |= 1
        } else {
            self.0 .1 &= usize::MAX - 1;
        }
    }

    //pub fn get_hap_id(&self) -> usize {
    //self.0 .2
    //}
}
