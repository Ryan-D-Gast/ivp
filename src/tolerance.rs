//! Tolerance abstraction to allow scalar or vector tolerances

use crate::Float;

/// Tolerance enum to allow scalar or vector tolerances
/// using [`Into`] trait for easy conversion from `Float`, `[Float; N]`, or `Vec<Float>`
/// users do not need to know or worry this simply allows both
/// `Float` and `[Float; N]` to be passed in as arguments.
#[derive(Clone, Debug)]
pub enum Tolerance {
    Scalar(Float),
    Vector(Vec<Float>),
}

impl From<Float> for Tolerance {
    fn from(val: Float) -> Self {
        Tolerance::Scalar(val)
    }
}

impl<const N: usize> From<[Float; N]> for Tolerance {
    fn from(val: [Float; N]) -> Self {
        Tolerance::Vector(val.to_vec())
    }
}

impl From<Vec<Float>> for Tolerance {
    fn from(val: Vec<Float>) -> Self {
        Tolerance::Vector(val)
    }
}

impl std::ops::Index<usize> for Tolerance {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            Tolerance::Scalar(v) => v,
            Tolerance::Vector(vs) => &vs[index],
        }
    }
}
