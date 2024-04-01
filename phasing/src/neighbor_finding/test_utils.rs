use crate::pbwt_trie::PbwtTrie;

pub fn gen_unique_haps_block(n_unique: usize, n_sites: usize, mut rng: impl rand::Rng) -> Vec<u64> {
    assert!(n_sites <= 64);
    let mask = u64::MAX >> (64 - n_sites);
    let mut unique = std::collections::HashSet::new();
    while unique.len() != n_unique {
        unique.insert(rng.gen::<u64>() & mask);
    }
    let mut haps = unique.into_iter().collect::<Vec<_>>();
    haps.sort();
    use rand::prelude::SliceRandom;
    haps.shuffle(&mut rng);
    haps
}

pub fn gen_index_map(n_unique: usize, multiplier: usize, mut rng: impl rand::Rng) -> Vec<u16> {
    let mut index_map = Vec::new();
    for _ in 0..multiplier {
        index_map.extend(0..n_unique as u16);
    }
    use rand::prelude::SliceRandom;
    index_map.shuffle(&mut rng);
    index_map
}

pub fn expand_ref(
    blocks: &[Vec<u64>],
    n_sites: &[usize],
    index_maps: &[Vec<u16>],
) -> Vec<Vec<bool>> {
    let n_haps = index_maps[0].len();
    let mut expanded = vec![Vec::<bool>::new(); n_haps];
    for ((block, &n_sites), index_map) in blocks
        .into_iter()
        .zip(n_sites.into_iter())
        .zip(index_maps.into_iter())
    {
        for site in hap_iter(block, n_sites) {
            for (&i, h) in index_map.into_iter().zip(&mut expanded) {
                h.push(site[i as usize]);
            }
        }
    }
    expanded
}

pub fn hap_iter(block: &[u64], n_sites: usize) -> impl Iterator<Item = Vec<bool>> + '_ {
    (0..n_sites).map(|site_i| {
        let mut site_hap = vec![false; block.len()];
        for (&b, t) in block.iter().zip(site_hap.iter_mut()) {
            *t = ((b >> site_i) & 1) != 0;
        }
        site_hap
    })
}

pub fn expand_pbwt_trie(pbwt_trie: &PbwtTrie) -> Vec<Vec<bool>> {
    let mut expanded: Vec<Vec<bool>> = vec![Vec::with_capacity(0)];
    for (layer, div) in pbwt_trie.trie.iter().zip(pbwt_trie.div.iter()) {
        let mut new_expanded: Vec<Vec<bool>> = vec![Vec::with_capacity(0); div.len()];
        for (node, h) in layer.into_iter().zip(expanded.iter()) {
            if let Some(pos) = node.0 {
                let mut h = h.clone();
                h.push(false);
                new_expanded[pos as usize] = h;
            }
            if let Some(pos) = node.1 {
                let mut h = h.clone();
                h.push(true);
                new_expanded[pos as usize] = h;
            }
        }
        expanded = new_expanded;
    }
    let mut out_expanded = vec![Vec::with_capacity(0); pbwt_trie.n_unique()];
    for (h, &id) in expanded.into_iter().zip(pbwt_trie.last_hap_ids.iter()) {
        out_expanded[id as usize] = h;
    }
    out_expanded
}

pub fn compute_div(a: &[bool], b: &[bool]) -> usize {
    assert_eq!(a.len(), b.len());
    for (i, (&a_, &b_)) in a.iter().zip(b.into_iter()).enumerate().rev() {
        if a_ != b_ {
            return i + 1;
        }
    }
    return 0;
}
pub fn compute_rev_prefix_match(a: &[bool], b: &[bool]) -> usize {
    assert_eq!(a.len(), b.len());
    let mut n = 0;
    for (&a_, &b_) in a.iter().zip(b.into_iter()).rev() {
        if a_ != b_ {
            break;
        }
        n += 1;
    }
    return n;
}

pub fn sort_by_reverse_prefix(haps: &mut [(usize, Vec<bool>)], site_i: usize) {
    haps.sort_by(|a, b| reverse_prefix_ord(&a.1, &b.1, site_i));
}

pub fn reverse_prefix_ord(a: &[bool], b: &[bool], site_i: usize) -> std::cmp::Ordering {
    let mut ord = std::cmp::Ordering::Equal;
    for (i, j) in a
        .iter()
        .take(site_i + 1)
        .zip(b.iter().take(site_i + 1))
        .rev()
    {
        if i != j {
            ord = i.cmp(j);
            break;
        }
    }
    ord
}

pub fn find_input_pos(input: &[bool], haps: &[Vec<bool>], site_i: usize) -> (usize, usize, usize) {
    let mut haps = haps.to_owned();
    haps.sort_by(|a, b| reverse_prefix_ord(a, b, site_i));

    let mut input_pos = haps.len();
    for (i, h) in haps.iter().enumerate() {
        use std::cmp::Ordering;
        match reverse_prefix_ord(input, h, site_i) {
            Ordering::Equal | Ordering::Less => {
                input_pos = i;
                break;
            }
            _ => {}
        }
    }

    let div_above = if input_pos == 0 {
        site_i + 1
    } else {
        let h = &haps[input_pos - 1];
        compute_div(
            &h[..(site_i + 1).min(h.len())],
            &input[..(site_i + 1).min(input.len())],
        )
    };

    let div_below = if input_pos == haps.len() {
        site_i + 1
    } else {
        let h = &haps[input_pos];
        compute_div(
            &h[..(site_i + 1).min(h.len())],
            &input[..(site_i + 1).min(input.len())],
        )
    };
    (input_pos, div_above, div_below)
}

pub fn compute_input_div(input: &[bool], haps: &[Vec<bool>], site_i: usize) -> Vec<usize> {
    haps.iter()
        .map(|h| compute_div(&h[..site_i.min(h.len())], &input[..site_i.min(input.len())]))
        .collect::<Vec<_>>()
}

pub use init::{init, init_first};

mod init {
    use super::*;
    pub fn init_first(genotype: i8) -> (bool, bool) {
        match genotype {
            1 => (true, false),
            0 => (false, false),
            2 => (true, true),
            _ => panic!("This shouldn't happen"),
        }
    }

    pub fn init(
        genotypes: &[i8],
        h_0: &mut [bool],
        h_1: &mut [bool],
        haps: &mut Vec<Vec<bool>>,
        site_i: usize,
    ) {
        pbwt_sort(haps, site_i);
        let (next_h_0, next_h_1) = if genotypes[site_i + 1] != 1 {
            match genotypes[site_i + 1] {
                0 => (false, false),
                2 => (true, true),
                _ => panic!("This shouldn't happen"),
            }
        } else {
            //let mut haps = haps.to_owned();
            //haps.sort_by(|a, b| reverse_prefix_ord(a, b, site_i));

            let init_values_0 = get_hap_div(&h_0, &haps, site_i);
            let init_values_1 = get_hap_div(&h_1, &haps, site_i);

            init_single_site_from_ranks(&init_values_0, &init_values_1)
        };
        h_0[site_i + 1] = next_h_0;
        h_1[site_i + 1] = next_h_1;
    }

    fn pbwt_sort(haps: &mut Vec<Vec<bool>>, site_i: usize) {
        let mut new_haps_0 = Vec::new();
        let mut new_haps_1 = Vec::new();

        for h in haps.drain(..) {
            if h[site_i] {
                new_haps_1.push(h);
            } else {
                new_haps_0.push(h);
            }
        }
        new_haps_0.extend(new_haps_1.into_iter());
        *haps = new_haps_0;
    }

    fn init_single_site_from_ranks(nn_0: &[(usize, bool)], nn_1: &[(usize, bool)]) -> (bool, bool) {
        let s = score(nn_0) - score(nn_1);

        if s.abs() >= 2 {
            if s > 0 {
                (true, false)
            } else {
                (false, true)
            }
        } else {
            if score_div(nn_0) > score_div(nn_1) {
                (true, false)
            } else {
                (false, true)
            }
        }
    }

    fn score(nn: &[(usize, bool)]) -> isize {
        nn.iter().map(|v| (v.1 as isize * 2 - 1)).sum()
    }

    fn score_div(rank: &[(usize, bool)]) -> f64 {
        rank.iter()
            .map(|v| (v.1 as isize * 2 - 1) as f64 * ((v.0 + 1) as f64).ln())
            .sum()
    }

    pub fn get_hap_div(input: &[bool], haps: &[Vec<bool>], site_i: usize) -> Vec<(usize, bool)> {
        let mut input_pos = haps.len();
        for (i, h) in haps.iter().enumerate() {
            use std::cmp::Ordering;
            match reverse_prefix_ord(input, h, site_i) {
                Ordering::Equal | Ordering::Less => {
                    input_pos = i;
                    break;
                }
                _ => {}
            }
        }

        let nn_0 = if input_pos == 0 { 1 } else { input_pos - 1 };

        let nn_1 = if input_pos == haps.len() {
            input_pos - 1
        } else {
            input_pos
        };

        let h_0 = haps[nn_0][site_i + 1];
        let h_1 = haps[nn_1][site_i + 1];
        let div_0 = compute_rev_prefix_match(
            &haps[nn_0][..(site_i + 1).min(input.len())],
            &input[..(site_i + 1).min(input.len())],
        );
        let div_1 = compute_rev_prefix_match(
            &haps[nn_1][..(site_i + 1).min(input.len())],
            &input[..(site_i + 1).min(input.len())],
        );

        vec![(div_0, h_0), (div_1, h_1)]
    }
}

pub fn display_hap(hap: &[bool]) -> String {
    hap.iter()
        .map(|&v| (v as u8).to_string())
        .reduce(|acc, x| acc + &x)
        .unwrap()
}

pub fn display_u64_hap(hap: u64, n_sites: usize) -> String {
    format!("{:0n_sites$b}", (hap << (64 - n_sites)).reverse_bits())
}
