use crate::neighbor_finding::pbwt_trie::{insert::Insert, Node};
use obliv_utils::vec::OblivVec;
use timing_shield::{TpBool, TpOrd, TpU16};

pub struct OblivInsert {
    zero: Vec<OblivVec<Wrapper>>,
    one: Vec<OblivVec<Wrapper>>,
}

impl OblivInsert {
    pub fn build(tree: &[Vec<Node>], div: &[Vec<u16>], n_zeros: &[u16]) -> Self {
        let (zero, one) = Insert::build(tree, div, n_zeros).deconstruct();
        let obliv_zero_all = zero
            .into_iter()
            .map(|z| {
                let mut obliv_zero = OblivVec::with_capacity(z.len());
                for (id, above, below) in z {
                    obliv_zero.push(Wrapper((
                        TpU16::protect(id),
                        TpU16::protect(above),
                        TpU16::protect(below),
                    )));
                }

                obliv_zero
            })
            .collect::<Vec<_>>();

        let obliv_one_all = one
            .into_iter()
            .map(|o| {
                let mut obliv_one = OblivVec::with_capacity(o.len());
                for (id, above, below) in o {
                    obliv_one.push(Wrapper((
                        TpU16::protect(id),
                        TpU16::protect(above),
                        TpU16::protect(below),
                    )));
                }
                obliv_one
            })
            .collect::<Vec<_>>();
        Self {
            zero: obliv_zero_all,
            one: obliv_one_all,
        }
    }

    pub fn insert<'a, 'b>(
        &'a self,
        hap_iter: impl Iterator<Item = TpBool> + 'b,
    ) -> impl Iterator<Item = (TpU16, TpU16, TpU16)> + 'a
    where
        'b: 'a,
    {
        let mut cur_group_id = TpU16::protect(0);
        let mut cur_div_above = TpU16::protect(0);
        let mut cur_div_below = TpU16::protect(0);
        hap_iter
            .zip(self.one.iter())
            .zip(self.zero.iter())
            .map(move |((h, one), zero)| {
                let mut zero = zero.to_owned();
                zero.cond_copy_from(one, h);
                let Wrapper((a, b, c)) = zero.get(cur_group_id.as_u32());
                cur_group_id = a;
                cur_div_above = cur_div_above.tp_gt(&b).select(cur_div_above, b);
                cur_div_below = cur_div_below.tp_gt(&c).select(cur_div_below, c);
                (cur_group_id, cur_div_above, cur_div_below)
            })
    }
}

#[derive(Clone)]
pub struct Wrapper((TpU16, TpU16, TpU16));

use timing_shield::TpCondSwap;

impl TpCondSwap for Wrapper {
    fn tp_cond_swap(cond: TpBool, a: &mut Self, b: &mut Self) {
        cond.cond_swap(&mut a.0 .0, &mut b.0 .0);
        cond.cond_swap(&mut a.0 .1, &mut b.0 .1);
        cond.cond_swap(&mut a.0 .2, &mut b.0 .2);
    }
}
