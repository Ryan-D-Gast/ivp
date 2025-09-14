//! Settings for numerical integrators

use bon::Builder;
use std::ops::{Index, IndexMut};

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
    /// Step-size strategy: predictive (Gustafsson) vs classical.
    /// - `true`  → 1: Modified predictive controller (Gustafsson) [default]
    /// - `false` → 2: Classical step-size control
    pub predictive: Option<bool>,
    /// Differential-Algebraic partitioning: counts of variables by index (1→2→3).
    ///
    /// - Variables must be ordered in the state as: all index-1 (differential),
    ///   then index-2 (algebraic), then index-3 (algebraic).
    /// - You can specify any subset of `nind1`, `nind2`, `nind3`:
    ///   - If none are provided, the system is treated as a pure ODE (all index-1).
    ///   - If `nind2`/`nind3` are provided but `nind1` is not, `nind1` is inferred as
    ///     `n - nind2 - nind3` (validated to be >= 0 at runtime).
    ///   - If all three are provided, they must sum to `n`.
    /// - Error estimation follows Radau5: index-2 contributions are multiplied by `h`,
    ///   and index-3 contributions by `h^2`.
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
