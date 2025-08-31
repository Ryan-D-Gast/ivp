//! Dormand-Prince Runge Kutta methods

mod dop853;
mod dopri5;
mod hinit;

use hinit::hinit;

pub use dop853::dop853;
pub use dopri5::dopri5;

use crate::{Float, status::Status};

#[derive(Clone, Debug)]
/// Settings for the Dormand-Prince integrators
///
/// # Settings
/// - `uround`  — rounding unit. Default ~ 2.3e-16.
/// - `safety_factor` — safety factor for step-size prediction.
///   Default 0.9.
/// - `fac` — lower/upper bounds for step-size ratio `hnew / hold`. Constrains
///   `fac.0 <= hnew/hold <= fac.1`. Default `(0.333..., 6.0)`.
/// - `beta` — stabilization parameter for step-size control.
///   Positive values (<= 0.04) stabilize control; negative inputs are treated
///   as zero. Default `0.04`.
/// - `h_max` — maximal step size; default is `xend - x0`.
/// - `h0` — initial step size; `None` triggers the
///   `hinit` heuristic to compute a starting guess.
/// - `nmax` — maximal number of allowed steps; default
///   `100_000`.
/// - `nstiff` — controls when the stiffness test is activated; default `1000`.
///
pub struct DPSettings {
    /// The rounding unit, typically machine epsilon
    pub uround: Float,
    /// safety factor in step-size prediction. Default is 0.9.
    pub safety_factor: Float,
    /// Parameter for step size selection where hfacl <= hnew/hold <= hfacu
    /// Default is (0.2, 10.0).
    pub fac: (Float, Float),
    /// Beta factor for stabilized step size control. Positive values of Beta
    /// ( <= 0.04 ) make the step size control more stable. Negative values
    /// are not accepted. Default is 0.04.
    pub beta: Float,
    /// Maximal step size. Default is xend - x0.
    pub h_max: Option<Float>,
    /// Initial step size. None will result in an initial guess
    /// provided by the [`hinit`] function.
    pub h0: Option<Float>,
    /// Maximum number of allowed steps. Default is 100,000.
    pub nmax: usize,
    /// Number of steps before performing a stiffness test. Default is 1000.
    pub nstiff: usize,
}

impl DPSettings {
    pub fn dopri5() -> Self {
        Self {
            uround: 2.3e-16,
            safety_factor: 0.9,
            fac: (0.2, 10.0),
            beta: 0.04,
            h_max: None,
            h0: None,
            nmax: 100_000,
            nstiff: 1000,
        }
    }

    pub fn dop853() -> Self {
        Self {
            uround: 2.3e-16,
            safety_factor: 0.9,
            fac: (0.3333333333333333, 6.0),
            beta: 0.0,
            h_max: None,
            h0: None,
            nmax: 100_000,
            nstiff: 1000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DPResult {
    pub x: Float,
    pub y: Vec<Float>,
    pub h: Float,
    pub status: Status,
    pub nfev: usize,
    pub nstep: usize,
    pub naccpt: usize,
    pub nrejct: usize,
}
