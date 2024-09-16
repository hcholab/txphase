use rust_htslib::bcf::record;

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
