use ndarray::ArrayView2;
use crate::pbwt::PBWT;

pub fn initialize(x: ArrayView2<u8>, t: &[i8], missing_bitmask: &[bool]) -> (Vec<u8>, Vec<u8>) {
    let mut pbwt = PBWT::new(x);
    let mut prev_col = pbwt.get_init_col().unwrap();
    let mut t_iter = t.iter();

    let mut s = 0.0; // score
    for (&is_missing, x_site) in missing_bitmask.iter().zip(x.rows().into_iter()) {
        if is_missing {

        } else {
            let site = *t_iter.next().unwrap();
            if site == 1 { // het site

            } else if site == -1 { // private missing site

            }

        }

    }
    todo!()
}
