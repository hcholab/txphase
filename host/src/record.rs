use rust_htslib::bcf::{self, record, Read, Reader};

#[repr(u8)]
#[derive(Clone, Copy, Eq, PartialEq, Debug, Hash)]
pub enum AlleleType {
    Ref,
    Alt,
}

impl From<i32> for AlleleType {
    fn from(genotype_index: i32) -> Self {
        match genotype_index {
            0 => Self::Ref,
            1 => Self::Alt,
            _ => panic!("invalid genotype"),
        }
    }
}

#[repr(u8)]
#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum TargetGenotype {
    Hom(AlleleType),
    Het,
    Missing,
}

impl TargetGenotype {
    pub fn is_missing(&self) -> bool {
        match self {
            Self::Missing => true,
            _ => false,
        }
    }

    pub fn is_het(&self) -> bool {
        match self {
            Self::Het => true,
            _ => false,
        }
    }
}

impl Into<i8> for TargetGenotype {
    fn into(self) -> i8 {
        match self {
            Self::Hom(t) => match t {
                AlleleType::Ref => 0,
                AlleleType::Alt => 2,
            },
            Self::Het => 1,
            Self::Missing => -1,
        }
    }
}

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub enum Genotype {
    Ref((AlleleType, AlleleType)),
    Target(TargetGenotype),
}

impl Genotype {
    pub fn is_ref(&self) -> bool {
        match self {
            Self::Ref(..) => true,
            Self::Target(..) => false,
        }
    }

    pub fn as_reference(self) -> Option<(AlleleType, AlleleType)> {
        if let Self::Ref(inner) = self {
            Some(inner)
        } else {
            None
        }
    }

    pub fn as_target(self) -> Option<TargetGenotype> {
        if let Self::Target(inner) = self {
            Some(inner)
        } else {
            None
        }
    }

    pub fn as_unphased(self) -> TargetGenotype {
        match self {
            Self::Ref((a, b)) => {
                if a == b {
                    TargetGenotype::Hom(a)
                } else {
                    TargetGenotype::Het
                }
            }
            Self::Target(t) => t,
        }
    }
}

impl From<record::Genotype> for Genotype {
    fn from(genotype: record::Genotype) -> Self {
        use record::GenotypeAllele;
        let v = &*genotype;
        assert_eq!(v.len(), 2);
        match v[0] {
            GenotypeAllele::Unphased(i) => match v[1] {
                GenotypeAllele::Phased(j) => Self::Ref((i.into(), j.into())),
                GenotypeAllele::Unphased(j) => Self::Target(if i == j {
                    TargetGenotype::Hom(i.into())
                } else {
                    TargetGenotype::Het
                }),
                _ => panic!("unsupported genotype"),
            },
            GenotypeAllele::UnphasedMissing => match v[1] {
                GenotypeAllele::UnphasedMissing => Self::Target(TargetGenotype::Missing),
                _ => panic!("unsupported genotype"),
            },
            _ => panic!("unsupported genotype"),
        }
    }
}

#[derive(Debug)]
pub struct Record {
    rid: u32,
    pos: i64,
    alleles: (String, String), // (REF, ALT)
    genotypes: Vec<Genotype>,
}

impl Record {
    fn from_bcf_record(mut bcf_record: bcf::Record) -> Option<Self> {
        bcf_record.unpack();
        if bcf_record.allele_count() != 2 {
            return None;
        }
        let rid = bcf_record.rid().unwrap();
        let pos = bcf_record.pos();
        let alleles = (
            String::from_utf8(bcf_record.alleles()[0].to_owned()).unwrap(),
            String::from_utf8(bcf_record.alleles()[1].to_owned()).unwrap(),
        );
        let sample_count = bcf_record.sample_count();
        let genotypes = {
            let genotypes = bcf_record.genotypes().unwrap();
            (0..sample_count)
                .map(|i| genotypes.get(i as usize).into())
                .collect::<Vec<_>>()
        };

        Some(Self {
            rid,
            pos,
            alleles,
            genotypes,
        })
    }
}

pub struct TargetRecord {
    pub sample_id: usize,
    pub sample_name: String,
    //pub chr: usize,
    pub positions: Vec<i64>,
    pub genotypes: Vec<TargetGenotype>, // (ref, alt)
}

impl TargetRecord {
    pub fn as_encoded_genotypes(&self) -> Vec<i8> {
        self.genotypes
            .iter()
            .map(|&v| match v {
                TargetGenotype::Hom(AlleleType::Ref) => 0,
                TargetGenotype::Hom(AlleleType::Alt) => 2,
                TargetGenotype::Het => 1,
                TargetGenotype::Missing => -1,
            })
            .collect()
    }
}

pub struct Records {
    sample_names: Vec<String>,
    records: Vec<Record>,
}

impl Records {
    pub fn from_path(path: impl AsRef<std::path::Path>) -> anyhow::Result<Self> {
        let mut reader = Reader::from_path(path)?;
        let mut records = Vec::new();
        let mut sample_names = None;

        for r in reader.records() {
            let r = r?;
            if sample_names.is_none() {
                sample_names = Some(
                    r.header()
                        .samples()
                        .into_iter()
                        .map(|s| String::from_utf8(s.to_owned()).unwrap())
                        .collect::<Vec<_>>(),
                );
            }
            if let Some(r) = Record::from_bcf_record(r) {
                records.push(r);
            }
        }
        Ok(Self {
            sample_names: sample_names.unwrap(),
            records,
        })
    }

    pub fn as_unphased_target_samples(&self) -> Vec<TargetRecord> {
        let mut target_records = vec![Vec::new(); self.sample_names.len()];
        let mut positions = Vec::new();

        for record in self.records.iter() {
            positions.push(record.pos);
            for (b, target_record) in record.genotypes.iter().zip(target_records.iter_mut()) {
                target_record.push(b.as_unphased());
            }
        }

        target_records
            .into_iter()
            .zip(self.sample_names.iter())
            .enumerate()
            .map(|(sample_id, (genotypes, sample_name))| TargetRecord {
                sample_id,
                sample_name: sample_name.clone(),
                genotypes,
                positions: positions.clone(),
            })
            .collect()
    }
}
