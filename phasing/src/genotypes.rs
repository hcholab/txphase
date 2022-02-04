const MIN_N_VARIANTS_PER_WINDOW: usize = 100;

pub struct GenotypesMeta {
}

impl GenotypesMeta {
    pub fn new() -> Self {
        Self {}
    }

    pub fn sub_genotype_meta(&self, start: usize, end: usize) -> GenotypesMetaView {
        GenotypesMetaView {
        }
    }
}

pub struct GenotypesMetaView {
}

impl GenotypesMetaView {
}
