//! Args for numerical integrators

use bon::Builder;

use crate::{Float, solout::SolOut};

#[derive(Builder)]
/// Args for the numerical integrators
/// 
/// Note if `solout` is not provided defining the type
/// as `Args` is required to use the default `DummySolOut`
/// e.g.
/// ```
/// let args: Args = ArgsBuilder::default().build();
/// //        ^- Without this will be unable to infer solout type.
/// ```
pub struct Args<S: SolOut> {
    /// Solution output function
    pub solout: Option<S>,
    /// Real tolerance for error estimation
    #[builder(default = 1e-6, into)]
    pub rtol: Tolerance,
    /// Absolute tolerance for error estimation
    #[builder(default = 1e-6, into)]
    pub atol: Tolerance,
    /// The rounding unit, typically machine epsilon
    #[builder(default = 2.3e-16)]
    pub uround: Float,
    /// safety factor in step-size prediction. Default is 0.9.
    #[builder(default = 0.9)]
    pub safety_factor: Float,
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
    /// Initial step size. None will result in an initial guess
    /// provided by the [`crate::hinit::hinit`] function.
    pub h0: Option<Float>,
    /// Maximum number of allowed steps. Default is 100,000.
    #[builder(default = 100_000)]
    pub nmax: usize,
    /// Number of steps before performing a stiffness test. Default is 1000.
    #[builder(default = 1000)]
    pub nstiff: usize,
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

impl std::ops::Index<usize> for Tolerance {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            Tolerance::Scalar(v) => v,
            Tolerance::Vector(vs) => &vs[index],
        }
    }
}
