use crate::pbwt_trie::Node;

pub struct Insert {
    zero: Vec<Vec<(u16, u16, u16)>>,
    one: Vec<Vec<(u16, u16, u16)>>,
}

impl Insert {
    pub fn build(trie: &[Vec<Node>], div: &[Vec<u16>], n_zeros: &[u16]) -> Self {
        let mut zero = Vec::with_capacity(trie.len());
        let mut one = Vec::with_capacity(trie.len());

        for (i, ((trie_level, div_level), &nz)) in trie
            .into_iter()
            .zip(std::iter::once(&vec![0]).chain(div.into_iter()))
            .zip(n_zeros.into_iter())
            .enumerate()
        {
            let (a, b) = Self::build_level(i as u16, trie_level, div_level, nz);
            zero.push(a);
            one.push(b);
        }
        Self { one, zero }
    }

    #[cfg(not(feature = "obliv"))]
    pub fn insert<'a, 'b>(
        &'a self,
        hap_iter: impl Iterator<Item = bool> + 'b,
    ) -> impl Iterator<Item = (u16, u16, u16)> + 'a
    where
        'b: 'a,
    {
        let mut cur_group_id = 0u16;
        let mut cur_div_above = 0u16;
        let mut cur_div_below = 0u16;
        hap_iter
            .zip(self.one.iter())
            .zip(self.zero.iter())
            .map(move |((h, one), zero)| {
                let (a, b, c) = if h {
                    one[cur_group_id as usize]
                } else {
                    zero[cur_group_id as usize]
                };
                cur_group_id = a;
                cur_div_above = cur_div_above.max(b);
                cur_div_below = cur_div_below.max(c);

                (cur_group_id, cur_div_above, cur_div_below)
            })
    }

    fn build_level(
        site_i: u16,
        trie_level: &[Node],
        div_level: &[u16],
        n_zeros: u16,
    ) -> (Vec<(u16, u16, u16)>, Vec<(u16, u16, u16)>) {
        let mut up_down = Vec::with_capacity(trie_level.len() + 1);
        let mut vq_down = Vec::with_capacity(trie_level.len() + 1);

        let mut p = site_i + 1;
        let mut q = site_i + 1;
        let mut u = 0;
        let mut v = n_zeros;

        up_down.push((u, p));
        vq_down.push((v, q));

        for (t, &d) in trie_level.iter().zip(div_level.iter()) {
            if p < d {
                p = d;
            }
            if q < d {
                q = d;
            }
            if let Some(u_) = t.0 {
                u = u_ + 1;
                p = 0;
            }
            if let Some(v_) = t.1 {
                v = v_ + 1;
                q = 0;
            }
            up_down.push((u, p));
            vq_down.push((v, q));
        }

        let mut p_up = Vec::with_capacity(trie_level.len() + 1);
        let mut q_up = Vec::with_capacity(trie_level.len() + 1);

        let mut p = site_i + 1;
        let mut q = site_i + 1;

        p_up.push(p);
        q_up.push(q);

        for (t, &d) in trie_level.iter().zip(div_level.iter()).rev() {
            if t.0.is_some() {
                p = 0;
            }
            if t.1.is_some() {
                q = 0;
            }

            p_up.push(p);
            q_up.push(q);

            if p < d {
                p = d;
            }
            if q < d {
                q = d;
            }
        }

        let zero = up_down
            .into_iter()
            .zip(p_up.into_iter().rev())
            .map(|((a, b), c)| (a, b, c))
            .collect();

        let one = vq_down
            .into_iter()
            .zip(q_up.into_iter().rev())
            .map(|((a, b), c)| (a, b, c))
            .collect();

        (zero, one)
    }

    #[cfg(feature = "obliv")]
    pub fn deconstruct(self) -> (Vec<Vec<(u16, u16, u16)>>, Vec<Vec<(u16, u16, u16)>>) {
        (self.zero, self.one)
    }
}
