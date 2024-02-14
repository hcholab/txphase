pub fn pbwt(
    next_site: usize,
    site_hap: &[bool],
    n_zeros: usize,
    cur_ppa: &[usize],
    cur_div: &[usize],
    next_ppa: &mut [usize],
    next_div: &mut [usize],
    mut save_permuted_hap: Option<&mut [bool]>,
) {
    //assert_eq!(site_hap.iter().filter(|&&v| !v).count(), n_zeros);
    //assert_eq!(cur_ppa.len(), site_hap.len());
    //assert_eq!(cur_div.len(), site_hap.len());
    //assert_eq!(cur_ppa.len(), next_ppa.len());
    //assert_eq!(cur_div.len(), next_div.len());

    if let Some(ref mut save_permuted_hap) = save_permuted_hap {
        save_permuted_hap.copy_from_slice(site_hap);
    }

    let mut u = 0;
    let mut v = n_zeros;
    let mut p = next_site;
    let mut q = next_site;
    for (&a, &d) in cur_ppa.iter().zip(cur_div.iter()) {
        if p < d {
            p = d;
        }
        if q < d {
            q = d;
        }
        if site_hap[a] {
            next_ppa[v] = a;
            next_div[v] = q;
            v = v + 1;
            q = 0;
        } else {
            next_ppa[u] = a;
            next_div[u] = p;
            u = u + 1;
            p = 0;
        }
    }
}
