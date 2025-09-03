//! Settings for numerical integrators

use std::marker::PhantomData;

use bon::Builder;

use crate::{Float, Tolerance};

#[derive(Builder)]
/// Settings for the numerical integrators
pub struct Settings<'a> {
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

    #[builder(default)]
    reference: PhantomData<&'a ()>
}