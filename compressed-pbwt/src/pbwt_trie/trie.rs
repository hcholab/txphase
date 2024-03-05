#[cfg(feature = "obliv")]
use crate::pbwt_trie::obliv_insert::OblivInsert;

#[cfg(not(feature = "obliv"))]
use crate::pbwt_trie::insert::Insert;

use crate::{Bool, U16};

pub type Node = (Option<u16>, Option<u16>);

pub type PosGroup = Vec<u16>;

pub struct PbwtTrie {
    pub start_site: usize,
    pub trie: Vec<Vec<Node>>,
    pub div: Vec<Vec<u16>>,
    pub first_haps: Vec<bool>,
    pub last_hap_ids: Vec<u16>,
    pub ppa: Vec<Vec<u32>>,
    #[cfg(feature = "obliv")]
    pub insert: OblivInsert,
    #[cfg(not(feature = "obliv"))]
    insert: Insert,
}

impl PbwtTrie {
    pub fn transform(
        start_site: usize,
        hap_site_iter: impl Iterator<Item = Vec<bool>>,
        prev_ppa: &[Vec<u32>],
        cur_index_map: &[u16],
        n_haps: usize,
        n_sites: usize,
    ) -> Self {
        assert!(n_haps <= u16::MAX as usize);
        assert!(n_sites <= u16::MAX as usize);

        let mut trie = Vec::new();
        let mut n_zeros = Vec::new();
        let mut cur_group_site = vec![(0..n_haps as u16).collect()];

        let mut div_all = Vec::new();
        let mut last_div = &vec![0];
        let mut first_haps = None;

        for (i, hap) in hap_site_iter.enumerate() {
            let (trie_layer, next_group_site, div, z) =
                build_trie_layer(&hap, i as u16, last_div, &cur_group_site);
            cur_group_site = next_group_site;
            trie.push(trie_layer);
            n_zeros.push(z);
            div_all.push(div);
            last_div = div_all.last().unwrap();
            if i == 0 {
                first_haps = Some(hap);
            }
        }
        let last_hap_ids = cur_group_site.iter().map(|v| v[0]).collect::<Vec<_>>();

        #[cfg(feature = "obliv")]
        let insert = OblivInsert::build(&trie, &div_all, &n_zeros);

        #[cfg(not(feature = "obliv"))]
        let insert = Insert::build(&trie, &div_all, &n_zeros);
        let ppa = build_ppa(prev_ppa, cur_index_map, &last_hap_ids);

        Self {
            start_site,
            trie,
            div: div_all,
            first_haps: first_haps.unwrap(),
            last_hap_ids,
            ppa,
            insert,
        }
    }

    pub fn insert<'a, 'b>(
        &'a self,
        hap_iter: impl Iterator<Item = Bool> + 'b,
    ) -> impl Iterator<Item = (U16, U16, U16)> + 'a
    where
        'b: 'a,
    {
        self.insert.insert(hap_iter)
    }

    pub fn n_sites(&self) -> usize {
        self.trie.len()
    }

    pub fn get_index_map(&self, n_haps: usize) -> Vec<u16> {
        let mut index_map = vec![0; n_haps];
        for (m, &i) in self.ppa.iter().zip(self.last_hap_ids.iter()) {
            for &j in m {
                index_map[j as usize] = i;
            }
        }
        index_map
    }

    pub fn n_unique(&self) -> usize {
        self.last_hap_ids.len()
    }
}

fn build_trie_layer(
    site_hap: &[bool],
    site_i: u16,
    cur_div: &[u16],
    cur_group_site: &[PosGroup],
) -> (Vec<Node>, Vec<PosGroup>, Vec<u16>, u16) {
    let mut p = site_i + 1;
    let mut q = site_i + 1;
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut d = Vec::new();
    let mut e = Vec::new();
    let mut trie_layer = Vec::new();
    for (&div_j, g) in cur_div.into_iter().zip(cur_group_site.into_iter()) {
        if p < div_j {
            p = div_j;
        }
        if q < div_j {
            q = div_j;
        }
        let mut node = Node::default();
        let mut group_a = None;
        let mut group_b = None;
        for &i in g {
            if site_hap[i as usize] {
                if node.1.is_none() {
                    node.1 = Some(e.len() as u16);
                    group_b = Some(Vec::new());
                    e.push(q);
                    q = 0;
                }
                group_b.as_mut().unwrap().push(i);
            } else {
                if node.0.is_none() {
                    node.0 = Some(d.len() as u16);
                    group_a = Some(Vec::new());
                    d.push(p);
                    p = 0;
                }
                group_a.as_mut().unwrap().push(i);
            }
        }
        if let Some(group_a) = group_a {
            a.push(group_a);
        }
        if let Some(group_b) = group_b {
            b.push(group_b);
        }
        trie_layer.push(node);
    }
    let z = d.len() as u16;
    a.extend(b.into_iter());
    d.extend(e.into_iter());
    for node in &mut trie_layer {
        if let Some(ref mut i) = node.1 {
            *i += z;
        }
    }
    (trie_layer, a, d, z)
}

fn build_ppa(prev_ppa: &[Vec<u32>], cur_index_map: &[u16], last_hap_ids: &[u16]) -> Vec<Vec<u32>> {
    let mut ppa = vec![Vec::new(); last_hap_ids.len()];
    let rev_last_hap_id = {
        let mut rev_last_hap_id = vec![0; last_hap_ids.len()];
        for (i, &j) in last_hap_ids.iter().enumerate() {
            rev_last_hap_id[j as usize] = i;
        }
        rev_last_hap_id
    };
    for &i in prev_ppa.into_iter().flatten() {
        ppa[rev_last_hap_id[cur_index_map[i as usize] as usize]].push(i);
    }
    ppa
}
