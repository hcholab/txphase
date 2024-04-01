use crate::{U16, U32};

#[cfg(feature = "obliv")]
use timing_shield::{TpEq, TpOrd};

pub struct PbwtTrieInput {
    pub full_div: Vec<U32>,
}

impl PbwtTrieInput {
    pub fn new(n_haps: usize) -> Self {
        #[cfg(feature = "obliv")]
        {
            Self {
                full_div: vec![U32::protect(0); n_haps],
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
        ppa: &[Vec<u32>],
    ) {
        expand_input_div(
            pos,
            div_above,
            div_below,
            last_trie_div,
            trie_start_site,
            ppa,
            &mut self.full_div,
        );
    }
}

fn expand_input_div(
    input_pos: U16,
    div_above: U16,
    div_below: U16,
    last_div: &[u16],
    start_site_i: usize,
    ppa: &[Vec<u32>],
    prev_div: &mut [U32],
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
            prev_div[i as usize] = d.tp_eq(&0).select(
                prev_div[i as usize],
                U32::protect(start_site_i as u32) + d.as_u32(),
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
