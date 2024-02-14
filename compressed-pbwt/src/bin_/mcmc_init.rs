use compressed_pbwt::site::Site;
use compressed_pbwt::*;
use rust_htslib::bcf;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::time::Instant;

#[cfg(feature = "obliv")]
use timing_shield::TpI8;

pub fn main() {
    let ref_panel_path = "data/data/1kg/m3vcf/20.1000g.Phase3.v5.With.Parameter.Estimates.m3vcf.gz";
    let ref_sites_path = "data/data/1kg/sites/chr20.csv";
    let input_path = "data/data/giab/HG002/chr/HG002_GRCh37_1_22_v4.2.1_benchmark_20.vcf.gz";
    //let output_path = "output/output.vcf.gz";
    //let parent1_path =
        //"data/data/giab/HG002/parent1/chr/HG003_GRCh37_1_22_v4.2.1_benchmark_20.vcf.gz";
    //let parent2_path =
        //"data/data/giab/HG002/parent2/chr/HG004_GRCh37_1_22_v4.2.1_benchmark_20.vcf.gz";

    eprintln!("### Reading Sites ... ###");
    let sites = site::sites_from_csv_path(&Path::new(ref_sites_path)).unwrap();

    eprintln!("### Reading Reference Panel ... ###");
    let (ref_panel_meta, ref_panel_block_iter) =
        m3vcf::load_ref_panel(std::path::Path::new(ref_panel_path));

    let ref_panel_blocks = ref_panel_block_iter.collect::<Vec<_>>();

    eprintln!("### Reading Input ... ###");
    let (
        mut target_samples,
        sites_bitmask,
        _input_bcf_header,
        _input_records_filtered,
        overlap_sites,
    ) = process_input(&Path::new(input_path), &sites);

    let target_sample = target_samples.pop().unwrap();

    eprintln!("### Scanning Reference Panel ... ###");
    let (ref_panel_new, _afreqs) =
        common::ref_panel::m3vcf_scan(&ref_panel_meta, &ref_panel_blocks, &sites_bitmask);

    //eprintln!("### Running reference solution ###");

    //let mut full_ref_haps = vec![Vec::new(); ref_panel_meta.n_haps];
    //for block in &ref_panel_new.blocks {
    //for site in block.as_slice().iter() {
    //for (&i, h) in block.index_map.iter().zip(full_ref_haps.iter_mut()) {
    //h.push(site[i]);
    //}
    //}
    //}

    //let mut ref_h_0 = vec![false; target_sample.len()];
    //let mut ref_h_1 = vec![false; target_sample.len()];

    //(ref_h_0[0], ref_h_1[0]) = compressed_pbwt::test_utils::init_first(target_sample[0]);

    //for i in 0..target_sample.len() - 1 {
    //compressed_pbwt::test_utils::init(
    //&target_sample,
    //&mut ref_h_0,
    //&mut ref_h_1,
    //&mut full_ref_haps,
    //i,
    //);
    //}

    //eprintln!("### Checking Correctness ... ###");
    //for (&g, (&ref_h_0, &ref_h_1)) in target_sample.iter().zip(ref_h_0.iter().zip(ref_h_1.iter())) {
    //assert_eq!(g, ref_h_0 as i8 + ref_h_1 as i8)
    //}

    //eprintln!("### Checking SER ... ###");

    //let (n_switches, n_tested, ser, n_errors) = switch_error_rate(
    //&ref_h_0,
    //&ref_h_1,
    //&overlap_sites,
    //&Path::new(parent1_path),
    //&Path::new(parent2_path),
    //);
    //eprintln!(
    //"# of switches\t=\t{}\n# tested\t=\t{}\nSER\t\t=\t{:.2}%\n# of errors\t=\t{}",
    //n_switches,
    //n_tested,
    //ser * 100.,
    //n_errors
    //);

    eprintln!("### Building PBWT Tries ... ###");
    let t = Instant::now();
    let mut pbwts: Vec<crate::pbwt_trie::PbwtTrie> = Vec::new();

    ref_panel_new
        .blocks
        .iter()
        .zip(std::iter::once(0).chain(ref_panel_new.block_map.iter().cloned()))
        .for_each(|(block, start_site)| {
            let block = block.as_slice();
            let index_map = block
                .index_map
                .iter()
                .map(|&v| v as u16)
                .collect::<Vec<_>>();
            let pbwt = if let Some(prev_pbwt) = pbwts.last() {
                crate::pbwt_trie::PbwtTrie::transform(
                    start_site,
                    block.iter(),
                    &prev_pbwt.ppa,
                    &index_map,
                    block.n_unique(),
                    block.n_sites(),
                )
            } else {
                let ppa = vec![(0..ref_panel_meta.n_haps).collect::<Vec<_>>()];
                crate::pbwt_trie::PbwtTrie::transform(
                    start_site,
                    block.iter(),
                    &ppa,
                    &index_map,
                    block.n_unique(),
                    block.n_sites(),
                )
            };
            pbwts.push(pbwt);
        });
    println!("time: {} ms", (Instant::now() - t).as_millis());

    eprintln!("### MCMC Init ... ###");

    #[cfg(feature = "obliv")]
    let target_sample = target_sample
        .into_iter()
        .map(|v| TpI8::protect(v))
        .collect::<Vec<_>>();

    let t = Instant::now();
    let (h_0, h_1) =
        compressed_pbwt::mcmc_init::mcmc_init(&target_sample, &pbwts, ref_panel_new.n_haps);
    println!("time: {} ms", (Instant::now() - t).as_millis());

    //eprintln!("### Checking Correctness ... ###");
    //assert_eq!(h_0.len(), target_sample.len());

    //for (&g, (&h_0, &h_1)) in target_sample.iter().zip(h_0.iter().zip(h_1.iter())) {
        //#[cfg(feature = "obliv")]
        //assert_eq!(g.expose(), h_0.expose() as i8 + h_1.expose() as i8);

        //#[cfg(not(feature = "obliv"))]
        //assert_eq!(g, h_0 as i8 + h_1 as i8);
    //}

    //eprintln!("### Checking SER ... ###");
    //#[cfg(feature = "obliv")]
    //let (n_switches, n_tested, ser, n_errors) = {
        //let h_0 = h_0.iter().map(|v| v.expose()).collect::<Vec<_>>();
        //let h_1 = h_1.iter().map(|v| v.expose()).collect::<Vec<_>>();
        //switch_error_rate(
            //&h_0,
            //&h_1,
            //&overlap_sites,
            //&Path::new(parent1_path),
            //&Path::new(parent2_path),
        //)
    //};

    //#[cfg(not(feature = "obliv"))]
    //let (n_switches, n_tested, ser, n_errors) = switch_error_rate(
        //&h_0,
        //&h_1,
        //&overlap_sites,
        //&Path::new(parent1_path),
        //&Path::new(parent2_path),
    //);

    //eprintln!(
        //"# of switches\t=\t{}\n# tested\t=\t{}\nSER\t\t=\t{:.2}%\n# of errors\t=\t{}",
        //n_switches,
        //n_tested,
        //ser * 100.,
        //n_errors
    //);

    //write_vcf(
    //output_path,
    //&h_0,
    //&h_1,
    //&input_bcf_header,
    //&input_records_filtered,
    //);

    let windows = [
        (0, 2335),
        (2335, 4955),
        (4955, 6979),
        (6979, 7948),
        (7948, 12153),
        (12153, 15706),
        (15706, 18661),
        (18661, 21284),
        (21284, 22644),
        (22644, 27057),
        (27057, 29619),
        (29619, 35931),
        (35931, 47577),
        (47577, 52598),
        (52598, 55539),
        (55539, 58561),
        (58561, 65062),
        (65062, 66992),
        (66992, 68558),
        (68558, 72989),
    ];

    eprintln!("### Finding Neighbors... ###");

    let t = Instant::now();
    let test_nn_vec = nn::find_top_neighbors(&h_0, 4, &pbwts, ref_panel_meta.n_haps, &vec![true; h_0.len()]);
    println!("time: {} ms", (Instant::now() - t).as_millis());

    //println!("Insert: {} ms", compressed_pbwt::nn::INSERT_T.lock().unwrap().as_millis());
    //println!("Init: {} ms", compressed_pbwt::nn::INIT_T.lock().unwrap().as_millis());
    //println!("Neighbors: {} ms", compressed_pbwt::nn::NEIGH_T.lock().unwrap().as_millis());
    //println!("Update: {} ms", compressed_pbwt::nn::UPDATE_T.lock().unwrap().as_millis());


    //#[cfg(feature = "obliv")]
    //let mut test_nn = test_nn.iter().map(|v| v.expose()).collect::<Vec<_>>();

    //for &(start, end) in &windows {
        //#[cfg(feature = "obliv")]
        //{
            //let mut set = obliv_utils::bitmap::OblivBitmap::new(ref_panel_meta.n_haps);
            //for i in test_nn_vec[start..end].iter().filter_map(|v| v.as_ref()).map(|v| v.iter()).flatten() {
                //set.set(i.as_u32());
            //}
            ////println!("{}", set.iter().filter(|v| v.expose()).count());
        //}

        //#[cfg(not(feature = "obliv"))]
        //{
            //let test_nn = test_nn_vec[start..end]
                //.iter()
                //.map(|v| v.into_iter())
                //.flatten()
                //.collect::<std::collections::HashSet<_>>();
            //println!("{}", test_nn.len());
        //}
    //}
}

//fn get_ref_neighbors(
//n_neighbors: usize,
//input_hap: &[bool],
//full_ref_haps: &[Vec<bool>],
//) -> Vec<Vec<HashSet<usize>>> {
//let mut all_neighbors = Vec::new();
//for i in 0..input_hap.len() {
//let input_hap = if i == input_hap.len() - 1 {
//&input_hap
//} else {
//&input_hap[..i + 1]
//};
//let mut ref_rank = HashMap::<usize, Vec<usize>>::new();
//for (j, h) in full_ref_haps.iter().enumerate() {
//let h = if i == h.len() - 1 { &h } else { &h[..i + 1] };
//let d = compute_div(input_hap, h);
//ref_rank
//.entry(d)
//.and_modify(|set| {
//set.push(j);
//})
//.or_insert(vec![j]);
//}
//let mut ref_rank = ref_rank.into_iter().collect::<Vec<_>>();
//ref_rank.sort_unstable();

//let mut neighbors = Vec::new();
//let mut n_neighbors = n_neighbors;
//for r in ref_rank {
//let set = r.1.iter().cloned().collect::<HashSet<usize>>();
//neighbors.push(set);
//n_neighbors -= n_neighbors.min(r.1.len());
//if n_neighbors == 0 {
//break;
//}
//}
//all_neighbors.push(neighbors);
//}
//all_neighbors
//}

//fn compute_div(a: &[bool], b: &[bool]) -> usize {
//assert_eq!(a.len(), b.len());
//for (i, (&a_, &b_)) in a.iter().zip(b.into_iter()).enumerate().rev() {
//if a_ != b_ {
//return i + 1;
//}
//}
//return 0;
//}

fn process_input(
    input_path: &Path,
    ref_sites: &[Site],
) -> (
    Vec<Vec<i8>>,
    Vec<bool>,
    bcf::header::HeaderView,
    Vec<bcf::record::Record>,
    Vec<Site>,
) {
    let (input_sites, input_bcf_header, input_bcf_records) =
        site::sites_from_bcf_path(input_path).unwrap();

    assert_eq!(input_sites.len(), input_bcf_records.len());

    let overlap = compute_overlap(&input_sites, &ref_sites);

    let n_overlap = overlap.len();
    let n_sites = ref_sites.len();

    let mut bcf_iter = input_sites.into_iter().zip(input_bcf_records.into_iter());
    let mut ref_sites_iter = ref_sites.into_iter();
    let mut ref_sites_bitmask = Vec::with_capacity(n_sites);

    let n_samples = input_bcf_header.sample_count() as usize;
    let mut samples: Vec<Vec<i8>> = vec![Vec::with_capacity(n_sites); n_samples];
    let mut input_records_filtered = Vec::with_capacity(n_overlap);

    for site in &overlap {
        while let Some(ref_site) = ref_sites_iter.next() {
            if *ref_site == *site {
                ref_sites_bitmask.push(true);
                break;
            } else {
                ref_sites_bitmask.push(false);
            }
        }

        while let Some((target_site, target_record)) = bcf_iter.next() {
            if target_site == *site {
                let genotypes = target_record.genotypes().unwrap();
                for (i, sample) in samples.iter_mut().enumerate() {
                    let genotype: record::Genotype = genotypes.get(i).into();
                    let genotype: i8 = genotype.as_unphased().into();
                    sample.push(genotype);
                }
                input_records_filtered.push(target_record);
                break;
            }
        }
    }

    while ref_sites_iter.next().is_some() {
        ref_sites_bitmask.push(false);
    }

    return (
        samples,
        ref_sites_bitmask,
        input_bcf_header,
        input_records_filtered,
        overlap,
    );
}

fn switch_error_rate(
    h0: &[bool],
    h1: &[bool],
    input_sites: &[Site],
    parent1_path: &Path,
    parent2_path: &Path,
) -> (usize, usize, f64, usize) {
    let (parent1_sites, _, _) = site::sites_from_bcf_path(parent1_path).unwrap();

    let (parent2_sites, _, _) = site::sites_from_bcf_path(parent2_path).unwrap();

    let overlap_sites = compute_overlap(
        &compute_overlap(&input_sites, &parent1_sites),
        &parent2_sites,
    );

    let mut site_map = Vec::new();
    let mut input_sites_iter = input_sites.iter().enumerate();

    for s in overlap_sites.iter() {
        while let Some((j, t)) = input_sites_iter.next() {
            if s == t {
                site_map.push(j);
                break;
            }
        }
    }

    let (parent1_sample, _, _, _, _) = process_input(&Path::new(parent1_path), &overlap_sites);
    let parent1_sample = &parent1_sample[0];
    let (parent2_sample, _, _, _, _) = process_input(&Path::new(parent2_path), &overlap_sites);
    let parent2_sample = &parent2_sample[0];

    let (h0, h1) = filter_overlap(&h0, &h1, input_sites, &overlap_sites);

    assert_eq!(h0.len(), parent1_sample.len());
    assert_eq!(h0.len(), parent2_sample.len());

    let mut trio_errors = 0;
    let mut n_switches = 0;
    let mut n_test = 0;
    let mut prev = -1;

    for (_, (((&h0, &h1), &p0), &p1)) in h0
        .iter()
        .zip(h1.iter())
        .zip(parent1_sample.iter())
        .zip(parent2_sample.iter())
        .enumerate()
    {
        if h0 == h1 {
            continue;
        }
        if p0 == 1 && p1 == 1 {
            continue;
        }

        if p0 == p1 {
            trio_errors += 1;
            continue;
        }

        let mut test_phase = 0;
        if p0 == 0 || p0 == 2 {
            test_phase = 1 + (h0 as i8 == p0 / 2) as i8;
        } else if p1 == 0 || p1 == 2 {
            test_phase = 1 + (h1 as i8 == p1 / 2) as i8;
        }

        if prev > 0 {
            if prev != test_phase {
                n_switches += 1;
            }
        };
        n_test += 1;
        prev = test_phase;
    }

    (
        n_switches,
        n_test,
        n_switches as f64 / n_test as f64,
        trio_errors,
    )
}

fn compute_overlap(a: &[Site], b: &[Site]) -> Vec<Site> {
    let a = a.iter().collect::<HashSet<_>>();
    let b = b.iter().collect::<HashSet<_>>();
    let mut overlap = a.intersection(&b).cloned().cloned().collect::<Vec<_>>();
    overlap.sort_unstable();
    overlap
}

fn filter_overlap(
    h0: &[bool],
    h1: &[bool],
    input_sites: &[Site],
    filter_sites: &[Site],
) -> (Vec<bool>, Vec<bool>) {
    let map = input_sites
        .into_iter()
        .cloned()
        .zip(h0.into_iter().cloned().zip(h1.into_iter().cloned()))
        .collect::<HashMap<Site, (bool, bool)>>();
    filter_sites
        .into_iter()
        .filter_map(|k| map.get(k).cloned())
        .unzip()
}

//fn write_vcf(
//file_name: &str,
//h_0: &[bool],
//h_1: &[bool],
//input_bcf_header: &bcf::header::HeaderView,
//input_records_filtered: &[bcf::record::Record],
//) {
//use bcf::record::GenotypeAllele;
//let phased = h_0
//.iter()
//.cloned()
//.zip(h_1.iter().cloned())
//.collect::<Vec<_>>();

//let mut out_vcf = bcf::Writer::from_path(
//&Path::new(file_name),
//&bcf::header::Header::from_template(&input_bcf_header),
//false,
//bcf::Format::Vcf,
//)
//.unwrap();

//for (genotype, input_record) in phased.into_iter().zip(input_records_filtered) {
//let mut new_record = out_vcf.empty_record();
//new_record.set_rid(input_record.rid());
//new_record.set_pos(input_record.pos());
//new_record.set_id(&input_record.id()).unwrap();
//new_record.set_alleles(&input_record.alleles()).unwrap();
//new_record
//.push_genotypes(&[
//GenotypeAllele::Phased(genotype.0 as i32),
//GenotypeAllele::Phased(genotype.1 as i32),
//])
//.unwrap();
//out_vcf.write(&new_record).unwrap();
//}
//}
