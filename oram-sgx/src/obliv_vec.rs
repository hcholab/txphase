//use crate::{ORAMBackend, ORAMBackendCreator, ORAM};
//use mc_oblivious_traits::subtle::Choice;
//use mc_oblivious_traits::CMov;

//pub struct OblivVec<C: ORAMBackendCreator, const VALUE_SIZE: usize> {
    //oram: ORAM<C, VALUE_SIZE>,
    //cur_len: u64,
    //capacity: u64,
//}

//impl<C: ORAMBackendCreator, const VALUE_SIZE: usize> OblivVec<C, VALUE_SIZE> {
    ///// does not automatically increase capacity
    //pub fn with_capacity(capacity: usize, oram_backend_creator: C) -> Self {
        //let oram = ORAM::new_empty(capacity + 1, oram_backend_creator);
        //Self {
            //oram,
            //cur_len: 0,
            //capacity: capacity as u64,
        //}
    //}

    //pub fn push(&mut self, elem: &[u8; VALUE_SIZE], choice: Choice) {
        //use mc_oblivious_traits::subtle::{ConditionallySelectable, ConstantTimeLess};
        //let has_space = self.cur_len.ct_lt(&self.capacity);
        //let i = u64::conditional_select(&self.capacity, &self.cur_len, choice & has_space);
        //self.oram.write(i as usize, elem);
        //self.cur_len =
            //u64::conditional_select(&self.cur_len, &(self.cur_len + 1), choice & has_space);
    //}

    //pub fn into_iter(self) -> Box<dyn Iterator<Item = [u8; VALUE_SIZE]>> {
        //Box::new(self.oram.into_iter().take(self.capacity as usize))
    //}

    //pub fn expose_len(&self) -> usize {
        //self.cur_len as usize
    //}
//}

//#[cfg(test)]
//mod test {
    //use super::*;

    //#[test]
    //fn obliv_vec() {
        //let capacity = 1000;
        //let mut obliv_vec = OblivVec::<_, 1>::with_capacity(capacity, crate::LeakyORAMCreator);
        //let mut choices = vec![0u8; capacity * 3];
        //use rand::RngCore;
        //rand::thread_rng().fill_bytes(&mut choices[..]);
        //let choices = choices
            //.into_iter()
            //.map(|b| Choice::from(b & 1))
            //.collect::<Vec<_>>();
        //let mut ref_vec = Vec::new();
        //for (i, choice) in choices.iter().enumerate() {
            //let v = (i % 256) as u8;
            //obliv_vec.push(&[v], *choice);
            //if choice.unwrap_u8() != 0 && ref_vec.len() < capacity {
                //ref_vec.push([v]);
            //}
        //}

        //let obliv_vec_len = obliv_vec.expose_len();
        //let vec_results = obliv_vec
            //.into_iter()
            //.take(obliv_vec_len)
            //.collect::<Vec<_>>();
        //println!("{}", vec_results.len());
        //assert_eq!(vec_results, ref_vec);
    //}
//}
