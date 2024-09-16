use crate::ref_panel::read_metadata;
use flate2::bufread::MultiGzDecoder;
use std::fs::File;
use std::io::Result;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(
    Debug, Eq, PartialEq, PartialOrd, Ord, Hash, Clone, serde::Serialize, serde::Deserialize,
)]
pub struct Site {
    pub chr: String,
    pub pos: u32,
    pub allele_ref: String,
    pub allele_alt: String,
}

pub fn read_sites(ref_panel_path: &Path) -> Vec<Site> {
    let f = File::open(ref_panel_path).expect("Unable to open reference file");
    let f = BufReader::new(f);
    let f = MultiGzDecoder::new(f);
    let f = BufReader::new(f);

    let mut lines_iter = f.lines();

    let ref_panel_meta = read_metadata(&mut lines_iter);
    let n_blocks = ref_panel_meta.n_blocks;
    let mut sites = Vec::new();

    for i in 0..n_blocks {
        let mut block_sites = read_block_sites(&mut lines_iter).unwrap();
        if i != n_blocks - 1 {
            block_sites.pop();
        }
        sites.extend(block_sites.into_iter());
    }
    sites
}

fn read_block_sites(mut lines_iter: impl Iterator<Item = Result<String>>) -> Option<Vec<Site>> {
    // read block metadata
    let line = lines_iter.next()?.unwrap();
    let mut iter = line.split_ascii_whitespace();
    let tok = iter.nth(7).unwrap(); // info field
    let tok = tok.split(";").collect::<Vec<_>>();

    let mut nvar = None;

    for t in tok {
        let t = t.split("=").collect::<Vec<_>>();
        match t[0] {
            "VARIANTS" => nvar = Some(t[1].parse::<usize>().unwrap()),
            _ => continue,
        }
    }

    let nvar = nvar.unwrap();

    let mut sites = Vec::with_capacity(nvar);

    iter.next().unwrap(); // skip one column

    // read block data
    for _ in 0..nvar {
        let line = lines_iter.next().unwrap().unwrap();
        let mut iter = line.split_ascii_whitespace();

        let site = {
            let chr = iter.next().unwrap().to_owned();
            let pos = iter.next().unwrap().parse::<u32>().unwrap();
            iter.next();
            let allele_ref = iter.next().unwrap().to_owned();
            let allele_alt = iter.next().unwrap().to_owned();
            Site {
                chr,
                pos,
                allele_ref,
                allele_alt,
            }
        };
        sites.push(site);
    }

    Some(sites)
}
