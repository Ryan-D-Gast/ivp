//! Settings for numerical integrators

use std::ops::{Index, IndexMut};
use bon::Builder;

use crate::{Float, matrix::MatrixStorage};


#[derive(Builder)]
/// Settings for the numerical integrators
pub struct Settings {
    /// The rounding unit, typically machine epsilon
    pub uround: Option<Float>,
    /// safety factor in step-size prediction.
    pub safety_factor: Option<Float>,
    /// Parameter for step size selection where scale_min <= hnew/hold <= scale_max
    pub scale_min: Option<Float>,
    /// Parameter for step size selection where scale_min <= hnew/hold <= scale_max
    pub scale_max: Option<Float>,
    /// Beta factor for stabilized step size control. Positive values of Beta
    /// ( <= 0.04 ) make the step size control more stable. Negative values
    /// are not accepted.
    pub beta: Option<Float>,
    /// Maximal step size.
    pub hmax: Option<Float>,
    /// Minimum step size.
    pub hmin: Option<Float>,
    /// Initial step size. None will result in an initial guess
    /// provided by the [`crate::hinit::hinit`] function.
    pub h0: Option<Float>,
    /// Maximum number of allowed steps.
    pub nmax: Option<usize>,
    /// Number of steps before performing a stiffness test.
    pub nstiff: Option<usize>,
    /// Max number of iterations in Newton solver.
    pub newton_maxiter: Option<usize>,
    /// Newton iteration tolerance.
    pub newton_tol: Option<Float>,

    /// Treat system as semi-explicit DAE with index-2 variables present.
    /// If `true`, variables in the ranges defined by `nind*` will be scaled
    /// according to Radau IIA index-handling rules.
    pub index2: Option<bool>,
    /// Treat system as semi-explicit DAE with index-3 variables present.
    /// If `true`, variables in the ranges defined by `nind*` will be scaled
    /// according to Radau IIA index-handling rules.
    pub index3: Option<bool>,
    /// Number of differential (index-1) variables at the start of the state vector.
    pub nind1: Option<usize>,
    /// Number of algebraic index-2 variables following the first block.
    pub nind2: Option<usize>,
    /// Number of algebraic index-3 variables following the first two blocks.
    pub nind3: Option<usize>,

    /// Preferred storage for the user-supplied Jacobian `jac(x,y,J)`.
    /// Default: `MatrixStorage::Full` (dense writable)
    #[builder(default = MatrixStorage::Full)]
    pub jac_storage: MatrixStorage,
    /// Preferred storage for the user-supplied mass matrix `mass(M)`.
    /// Default: `MatrixStorage::Identity` (implicit identity)
    #[builder(default = MatrixStorage::Identity)]
    pub mass_storage: MatrixStorage,
}

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

impl From<&[Float]> for Tolerance {
    fn from(val: &[Float]) -> Self {
        Tolerance::Vector(val.to_vec())
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

impl Index<usize> for Tolerance {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            Tolerance::Scalar(v) => v,
            Tolerance::Vector(vs) => &vs[index],
        }
    }
}

impl IndexMut<usize> for Tolerance {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self {
            Tolerance::Scalar(v) => v,
            Tolerance::Vector(vs) => &mut vs[index],
        }
    }
}
