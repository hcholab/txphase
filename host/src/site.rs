use m3vcf::Site;
use rust_htslib::bcf;
use std::path::Path;

//#[derive(serde::Deserialize, Debug, Eq, PartialEq, PartialOrd, Ord, Hash, Clone)]
//pub struct MySite {
    //#[serde(rename = "#CHROM")]
    //pub chr: String,
    //#[serde(rename = "POS")]
    //pub pos: u32,
    //#[serde(rename = "REF")]
    //pub allele_ref: String,
    //#[serde(rename = "ALT")]
    //pub allele_alt: String,
//}

pub fn sites_from_bcf_path(
    path: &Path,
) -> anyhow::Result<(Vec<Site>, bcf::header::HeaderView, Vec<bcf::record::Record>)> {
    let mut reader = bcf::Reader::from_path(path)?;
    reader.set_threads(10).unwrap();
    let mut records = Vec::new();
    use bcf::Read;
    let header = reader.header().clone();
    let mut sites = Vec::new();
    for record in reader.records() {
        let record = record?;
        if let Some(site) = site_from_bcf_record(&record, &header)? {
            sites.push(site);
            records.push(record);
        }
    }
    Ok((sites, header, records))
}

//pub fn sites_from_csv_path(path: &Path) -> anyhow::Result<Vec<Site>> {
    //let mut reader = csv::Reader::from_path(path)?;
    //let sites = reader
        //.deserialize()
        //.map(|r| r.unwrap())
        //.collect::<Vec<MySite>>();
    //let sites = sites
        //.into_iter()
        //.map(|site| Site {
            //chr: site.chr,
            //pos: site.pos,
            //allele_ref: site.allele_ref,
            //allele_alt: site.allele_alt,
        //})
        //.collect();
    //Ok(sites)
//}

fn site_from_bcf_record(
    record: &bcf::record::Record,
    header: &bcf::header::HeaderView,
) -> anyhow::Result<Option<Site>> {
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
    Ok(Some(Site {
        chr,
        pos,
        allele_ref,
        allele_alt,
    }))
}
