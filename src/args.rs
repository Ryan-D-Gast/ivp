//! Args for numerical integrators

use std::marker::PhantomData;

use bon::Builder;

use crate::{Float, solout::{SolOut, DummySolOut}};

#[derive(Builder)]
/// Args for the numerical integrators
pub struct Args<'a, S: SolOut = DummySolOut> {
    /// Solution output function
    pub solout: Option<S>,
    /// Real tolerance for error estimation
    #[builder(default = 1e-6, into)]
    pub rtol: Tolerance<'a>,
    /// Absolute tolerance for error estimation
    #[builder(default = 1e-6, into)]
    pub atol: Tolerance<'a>,
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

    /// Phantom data for lifetime tracking
    #[builder(default)]
    _phantom_reference: PhantomData<&'a ()>
}

/// Tolerance enum to allow scalar or vector tolerances
/// using [`Into`] trait for easy conversion from `Float`, `[Float; N]`, or `Vec<Float>`
/// users do not need to know or worry this simply allows both
/// `Float` and `[Float; N]` to be passed in as arguments.
#[derive(Clone, Debug)]
pub enum Tolerance<'a> {
    Scalar(Float),
    Vector(&'a [Float]),
}

impl From<Float> for Tolerance<'_> {
    fn from(val: Float) -> Self {
        Tolerance::Scalar(val)
    }
}

impl<'a> From<&'a [Float]> for Tolerance<'a> {
    fn from(val: &'a [Float]) -> Self {
        Tolerance::Vector(val)
    }
}

impl std::ops::Index<usize> for Tolerance<'_> {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            Tolerance::Scalar(v) => v,
            Tolerance::Vector(vs) => &vs[index],
        }
    }
}
