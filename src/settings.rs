//! Settings for numerical integrators

use crate::Float;

#[derive(Clone, Debug)]
/// Settings for the numerical integrators
pub struct Settings {
    /// The rounding unit, typically machine epsilon
    pub uround: Float,
    /// safety factor in step-size prediction. Default is 0.9.
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
    pub nmax: usize,
    /// Number of steps before performing a stiffness test. Default is 1000.
    pub nstiff: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            uround: 2.3e-16,
            safety_factor: 0.9,
            scale_min: None,
            scale_max: None,
            beta: None,
            hmax: None,
            h0: None,
            nmax: 100_000,
            nstiff: 1000,
        }
    }
}
