use crate::{Usize, U16};

#[cfg(feature = "obliv")]
use timing_shield::{TpEq, TpOrd};

pub struct PbwtTrieInput {
    pub full_pos: Usize,
    pub full_div: Vec<Usize>,
}

impl PbwtTrieInput {
    pub fn new(n_haps: usize) -> Self {
        #[cfg(feature = "obliv")]
        {
            Self {
                full_pos: Usize::protect(0),
                full_div: vec![Usize::protect(0); n_haps],
            }
        }

        #[cfg(not(feature = "obliv"))]
        {
            Self {
                full_pos: 0,
                full_div: vec![0; n_haps],
            }
        }
    }

    pub fn update(
        &mut self,
        pos: U16,
        div_above: U16,
        div_below: U16,
        trie_start_site: usize,
        last_trie_div: &[u16],
        prev_ppa: &[Vec<usize>],
        cur_ppa: &[Vec<usize>],
    ) {
        expand_input_div(
            pos,
            div_above,
            div_below,
            last_trie_div,
            trie_start_site,
            cur_ppa,
            &mut self.full_div,
        );

        self.full_pos = expand_input_ppa(
            pos,
            div_above,
            div_below,
            self.full_pos,
            prev_ppa,
            cur_ppa,
            self.full_div.len(),
        );
        //#[cfg(feature = "obliv")]
        //println!(
        //"input: {}, a: {}, b: {}",
        //self.full_pos.expose(),
        //div_above.expose(),
        //div_below.expose()
        //);
        //#[cfg(not(feature = "obliv"))]
        //println!(
        //"input: {}, a: {}, b: {}",
        //self.full_pos, div_above, div_below
        //);
    }
}

fn expand_input_div(
    input_pos: U16,
    div_above: U16,
    div_below: U16,
    last_div: &[u16],
    start_site_i: usize,
    ppa: &[Vec<usize>],
    prev_div: &mut [Usize],
) {
    #[cfg(feature = "obliv")]
    let mut input_div = vec![U16::protect(0); last_div.len()];

    #[cfg(feature = "obliv")]
    let last_div = last_div
        .iter()
        .map(|&v| U16::protect(v))
        .collect::<Vec<_>>();

    #[cfg(not(feature = "obliv"))]
    let mut input_div = vec![0; last_div.len()];

    #[cfg(feature = "obliv")]
    {
        // sweep up
        let mut d = div_above;
        for i in (1..input_div.len() as u16 + 1).rev() {
            let cond1 = input_pos.tp_eq(&i);
            let cond2 = input_pos.tp_gt(&i);
            let i = i as usize;

            let max_d = if i < input_div.len() {
                d.tp_gt(&last_div[i]).select(d, last_div[i])
            } else {
                d
            };

            d = cond2.select(max_d, d);

            input_div[i - 1] = (cond1 | cond2).select(d, input_div[i - 1]);
        }

        // sweep down
        let mut d = div_below;
        for i in 0..input_div.len() as u16 {
            let cond1 = input_pos.tp_eq(&i);
            let cond2 = input_pos.tp_lt(&i);
            let i = i as usize;

            let max_d = d.tp_gt(&last_div[i]).select(d, last_div[i]);
            d = cond2.select(max_d, d);

            input_div[i] = (cond1 | cond2).select(d, input_div[i]);
        }
    }

    #[cfg(not(feature = "obliv"))]
    {
        // sweep up
        if input_pos > 0 {
            let mut d = div_above;
            input_div[input_pos as usize - 1] = d;
            for i in (1..input_pos as usize).rev() {
                d = d.max(last_div[i]);
                input_div[i - 1] = d;
            }
        }

        // sweep down
        if (input_pos as usize) < last_div.len() {
            let mut d = div_below;
            input_div[input_pos as usize] = d;
            for i in input_pos as usize + 1..last_div.len() {
                d = d.max(last_div[i]);
                input_div[i] = d;
            }
        }
    }

    for (&d, group) in input_div.iter().zip(ppa.iter()) {
        #[cfg(feature = "obliv")]
        for &i in group {
            prev_div[i] = d.tp_eq(&0).select(
                prev_div[i],
                Usize::protect(start_site_i as u64) + d.as_u64(),
            );
        }

        #[cfg(not(feature = "obliv"))]
        if d != 0 {
            for &i in group {
                prev_div[i] = start_site_i as usize + d as usize;
            }
        }
    }
}

fn expand_input_ppa(
    input_pos: U16,
    div_above: U16,
    div_below: U16,
    prev_full_pos: Usize,
    prev_ppa: &[Vec<usize>],
    cur_ppa: &[Vec<usize>],
    n_haps: usize,
) -> Usize {
    #[cfg(feature = "obliv")]
    {
        // in-between groups
        let mut full_pos_a = Usize::protect(0);
        for (i, group) in cur_ppa.into_iter().enumerate() {
            full_pos_a = input_pos
                .tp_gt(&(i as u16))
                .select(full_pos_a + Usize::protect(group.len() as u64), full_pos_a);
        }

        // group member
        use timing_shield::TpI64;
        let mut order = vec![TpI64::protect(0); n_haps];
        for (i, &pos) in prev_ppa.iter().flatten().enumerate() {
            let i = TpI64::protect(i as i64);
            order[pos] = i
                .tp_gt_eq(&prev_full_pos.as_i64())
                .select(i - prev_full_pos.as_i64() + 1, i - prev_full_pos.as_i64());
        }
        let group_id = div_below.tp_eq(&0).select(input_pos, input_pos - 1);
        let mut full_pos_b = Usize::protect(0);

        for (i, group) in cur_ppa.into_iter().enumerate() {
            let i = i as u16;
            let ingroup_n = group.into_iter().fold(Usize::protect(0), |accu, &pos| {
                order[pos].tp_lt(&0).select(accu + 1, accu)
            });
            full_pos_b = group_id
                .tp_eq(&i)
                .select(full_pos_b + ingroup_n, full_pos_b);

            full_pos_b = group_id
                .tp_gt(&i)
                .select(full_pos_b + Usize::protect(group.len() as u64), full_pos_b);
        }

        let is_between_groups = (!div_above.tp_eq(&0)) & (!div_below.tp_eq(&0));

        is_between_groups.select(full_pos_a, full_pos_b)
    }

    #[cfg(not(feature = "obliv"))]
    {
        let mut cur_full_pos;
        if div_above != 0 && div_below != 0 {
            cur_full_pos = cur_ppa
                .iter()
                .map(|group| group.len())
                .take(input_pos as usize)
                .sum::<usize>();
        } else {
            let mut order = vec![0isize; n_haps];
            for (i, &pos) in prev_ppa.iter().flatten().enumerate() {
                order[pos] = if i >= prev_full_pos {
                    i as isize - prev_full_pos as isize + 1
                } else {
                    i as isize - prev_full_pos as isize
                };
            }
            let group_id = if div_below == 0 {
                input_pos
            } else {
                input_pos - 1
            } as usize;
            cur_full_pos = cur_ppa
                .iter()
                .map(|group| group.len())
                .take(group_id)
                .sum::<usize>();

            let mut found = false;
            for (i, &pos) in cur_ppa[group_id].iter().enumerate() {
                if order[pos] >= 0 {
                    cur_full_pos += i;
                    found = true;
                    break;
                }
            }
            if !found {
                cur_full_pos += cur_ppa[group_id].len();
            }
        }
        cur_full_pos
    }
}
