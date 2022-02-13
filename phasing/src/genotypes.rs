use crate::Genotype;
use ndarray::{Array1, ArrayView1};

pub const HET_PER_SEGMENT: usize = 3;
pub const P: usize = 1 << HET_PER_SEGMENT;

#[repr(i8)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum G {
    HomRef = 0,
    HomAlt = 2,
    Het1 = -1,
    Het2 = -2,
    Het3 = -3,
}

impl G {
    #[inline]
    pub fn access_graph_row(self, i: u8) -> i8 {
        assert!(i < P as u8);
        let g = self as i8;
        (if g < 0 {
            (i >> (-g as u8)) & 1
        } else {
            g as u8 / 2
        }) as i8
    }

    #[inline]
    pub fn is_segment_marker(self) -> bool {
        self == Self::Het3
    }
}

impl From<i8> for G {
    fn from(g: i8) -> Self {
        match g {
            0 => Self::HomRef,
            2 => Self::HomAlt,
            -1 => Self::Het1,
            -2 => Self::Het2,
            -3 => Self::Het3,
            _ => panic!("Invalid genotype"),
        }
    }
}

impl Into<i8> for G {
    fn into(self) -> i8 {
        self as i8
    }
}

pub struct Genotypes {
    pub genotypes: Array1<G>,
}

impl Genotypes {
    pub fn new(genotypes: ArrayView1<Genotype>) -> Self {
        let mut het_counter = 0i8;
        let mut inner = Vec::<G>::with_capacity(genotypes.len());
        for &g in genotypes {
            if g == 1 {
                het_counter += 1;
                inner.push((-het_counter).into());
                if het_counter == HET_PER_SEGMENT as i8 {
                    het_counter = 0;
                }
            } else {
                inner.push(g.into());
            }
        }
        Self {
            genotypes: Array1::from_vec(inner),
        }
    }

    #[inline]
    pub fn access_graph(&self, pos: usize, i: u8) -> i8 {
        self.genotypes[pos].access_graph_row(i)
    }
}
