use rust_htslib::bcf;
use std::collections::{HashMap, HashSet};
use std::path::Path;

fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    assert_eq!(args.len(), 4);

    println!("Samples: {}", &args[1]);
    println!("Parents: {}", &args[2]);
    println!("Trios: {}", &args[3]);

    let samples_path = Path::new(&args[1]);
    let parents_path = Path::new(&args[2]);
    let trios_path = Path::new(&args[3]);

    let (samples_sites, samples_sample_ids, samples_records) =
        sites_from_bcf_path(&samples_path).unwrap();
    let (parents_sites, parents_sample_ids, parents_records) =
        sites_from_bcf_path(&parents_path).unwrap();

    let trios = read_trios_from_path(&trios_path);

    let parents_id_mask = {
        let parents_sample_ids_set = samples_sample_ids
            .iter()
            .map(|v| trios[v].iter().cloned())
            .flatten()
            .collect::<HashSet<String>>();
        parents_sample_ids
            .iter()
            .map(|v| parents_sample_ids_set.contains(v))
            .collect::<Vec<_>>()
    };

    let parents_id_indices = parents_id_mask
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| if b { Some(i) } else { None })
        .collect::<Vec<_>>();

    let intersection = {
        let samples_sites_set = samples_sites.iter().cloned().collect::<HashSet<_>>();
        let parents_sites_set = parents_sites.iter().cloned().collect::<HashSet<_>>();
        samples_sites_set
            .intersection(&parents_sites_set)
            .cloned()
            .collect::<HashSet<Site>>()
    };

    let (samples_sites, samples_records): (Vec<_>, Vec<_>) = samples_sites
        .into_iter()
        .zip(samples_records.into_iter())
        .filter(|(s, _)| intersection.contains(s))
        .unzip();

    let (parents_sites, parents_records): (Vec<_>, Vec<_>) = parents_sites
        .into_iter()
        .zip(parents_records.into_iter())
        .filter(|(s, _)| intersection.contains(s))
        .unzip();

    assert_eq!(samples_sites, parents_sites);

    let mut parents_indiv_records = parents_sample_ids
        .iter()
        .zip(parents_id_mask.iter())
        .filter_map(|(id, &b)| if b { Some(id) } else { None })
        .map(|id| (id, Vec::new()))
        .collect::<Vec<_>>();

    for r in parents_records {
        let genotypes = r.genotypes().unwrap();
        parents_id_indices
            .iter()
            .zip(parents_indiv_records.iter_mut())
            .for_each(|(&i, v)| {
                v.1.push({
                    let g = genotypes.get(i);
                    (g[0], g[1])
                })
            });
    }

    let parents_indiv_records = parents_indiv_records.into_iter().collect::<HashMap<_, _>>();

    let mut samples_indiv_records = samples_sample_ids
        .into_iter()
        .map(|v| (v, Vec::new()))
        .collect::<Vec<_>>();

    for r in samples_records {
        let genotypes = r.genotypes().unwrap();
        for (i, s) in samples_indiv_records.iter_mut().enumerate() {
            s.1.push({
                let g = genotypes.get(i);
                (g[0], g[1])
            })
        }
    }

    for (sample_id, sample_genotypes) in samples_indiv_records {
        let [parent1_id, parent2_id] = trios[&sample_id].clone();
        let (n_test, n_switches, trio_errors) = switch_error_check(
            &sample_genotypes,
            &parents_indiv_records[&parent1_id],
            &parents_indiv_records[&parent2_id],
        );
        println!(
            "{sample_id}: {n_test}, {n_switches}, {trio_errors}, {:.3}",
            n_switches as f64 / n_test as f64 * 100.
        );
    }
}

#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct Site {
    pub chr: String,
    pub pos: u32,
    pub allele_ref: String,
    pub allele_alt: String,
}

impl Site {
    pub fn from_bcf_record(
        record: &bcf::record::Record,
        header: &bcf::header::HeaderView,
    ) -> anyhow::Result<Option<Self>> {
        if record.allele_count() != 2 {
            return Ok(None);
        }
        let chr = String::from_utf8(
            header
                .rid2name(record.rid().ok_or(anyhow::anyhow!("Missing rid"))?)?
                .to_owned(),
        )?;
        let pos = record.pos() as u32;
        let allele_ref = String::from_utf8(record.alleles()[0].to_owned())?;
        let allele_alt = String::from_utf8(record.alleles()[1].to_owned())?;
        Ok(Some(Self {
            chr,
            pos,
            allele_ref,
            allele_alt,
        }))
    }
}

pub fn sites_from_bcf_path(
    path: &Path,
) -> anyhow::Result<(Vec<Site>, Vec<String>, Vec<bcf::record::Record>)> {
    let mut reader = bcf::Reader::from_path(path)?;
    reader.set_threads(8).unwrap();
    let mut records = Vec::new();
    use bcf::Read;
    let header = reader.header().clone();
    let mut sites = Vec::new();
    for record in reader.records() {
        let record = record?;
        if let Some(site) = Site::from_bcf_record(&record, &header)? {
            sites.push(site);
            records.push(record);
        }
    }
    let sample_ids = header
        .samples()
        .into_iter()
        .map(|v| String::from_utf8(v.to_owned()).unwrap())
        .collect::<Vec<_>>();

    assert_eq!(sample_ids.len(), records[0].sample_count() as usize);
    Ok((sites, sample_ids, records))
}

fn read_trios_from_path(path: &Path) -> HashMap<String, [String; 2]> {
    use std::io::BufRead;
    let trios_file = std::io::BufReader::new(std::fs::File::open(path).unwrap());
    trios_file
        .lines()
        .map(|line| {
            let line = line.unwrap();
            let tokens = line.split_whitespace().collect::<Vec<_>>();
            let child = tokens[1].to_owned();
            let parent1 = tokens[2].to_owned();
            let parent2 = tokens[3].to_owned();
            (child, [parent1, parent2])
        })
        .collect()
}

use rust_htslib::bcf::record::GenotypeAllele;

fn switch_error_check(
    target_sample: &[(GenotypeAllele, GenotypeAllele)],
    parent1: &[(GenotypeAllele, GenotypeAllele)],
    parent2: &[(GenotypeAllele, GenotypeAllele)],
) -> (usize, usize, usize) {
    assert_eq!(target_sample.len(), parent1.len());
    assert_eq!(target_sample.len(), parent2.len());

    let mut trio_errors = 0;
    let mut n_switches = 0;
    let mut n_test = 0;
    let mut prev = -1;

    for (i, ((t, &p1), &p2)) in target_sample
        .iter()
        .zip(parent1.iter())
        .zip(parent2.iter())
        .enumerate()
    {
        let (h0, h1) = match t {
            (GenotypeAllele::Phased(h0), GenotypeAllele::Phased(h1)) => (*h0 as i8, *h1 as i8),
            (GenotypeAllele::Unphased(h0), GenotypeAllele::Phased(h1)) => (*h0 as i8, *h1 as i8),
            (GenotypeAllele::PhasedMissing, GenotypeAllele::PhasedMissing) => continue,
            (GenotypeAllele::UnphasedMissing, GenotypeAllele::UnphasedMissing) => continue,
            _ => panic!("This shouldn't happen"),
        };

        let p1 = match p1 {
            (GenotypeAllele::Unphased(a), GenotypeAllele::Unphased(b)) => a as i8 + b as i8,
            (GenotypeAllele::UnphasedMissing, GenotypeAllele::UnphasedMissing) => continue,
            _ => panic!("This shouldn't happen"),
        };

        let p2 = match p2 {
            (GenotypeAllele::Unphased(a), GenotypeAllele::Unphased(b)) => a as i8 + b as i8,
            (GenotypeAllele::UnphasedMissing, GenotypeAllele::UnphasedMissing) => continue,
            _ => panic!("This shouldn't happen"),
        };

        if h0 == h1 {
            continue;
        }

        // Ambiguous
        if p1 == 1 && p2 == 1 {
            continue;
        }

        // Mandelian error
        if p1 == p2 {
            trio_errors += 1;
            continue;
        }

        let mut test_phase = 0;
        if p1 == 0 || p1 == 2 {
            test_phase = 1 + (h0 == p1 / 2) as i8;
        } else if p2 == 0 || p2 == 2 {
            test_phase = 1 + (h1 == p2 / 2) as i8;
        }

        if prev > 0 {
            if prev != test_phase {
                println!("{n_test}, {i}: {test_phase}, {h0}, {h1}, {p1}, {p2}");
                n_switches += 1;
            }
        };
        n_test += 1;
        prev = test_phase;
    }
    (n_test, n_switches, trio_errors)
}
