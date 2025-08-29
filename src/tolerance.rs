//! Tolerance abstraction to allow scalar or vector tolerances

use crate::Float;

/// Tolerance enum to allow scalar or vector tolerances
/// using [`Into`] trait for easy conversion from `Float` or `[Float; N]`
/// users do not need to know or worry this simply allows both
/// `Float` and `[Float; N]` to be passed in as arguments.
#[derive(Clone, Copy, Debug)]
pub enum Tolerance<const N: usize> {
    Scalar(Float),
    Vector([Float; N]),
}

impl<const N: usize> From<Float> for Tolerance<N> {
    fn from(val: Float) -> Self {
        Tolerance::Scalar(val)
    }
}

impl<const N: usize> From<[Float; N]> for Tolerance<N> {
    fn from(val: [Float; N]) -> Self {
        Tolerance::Vector(val)
    }
}

impl<const N: usize> std::ops::Index<usize> for Tolerance<N> {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            Tolerance::Scalar(v) => v,
            Tolerance::Vector(vs) => &vs[index],
        }
    }
}