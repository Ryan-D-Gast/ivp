//! Options and method selection for solve_ivp

use bon::Builder;

use crate::{Float, methods::settings::Tolerance};

/// Solver method selection (roughly mirroring scipy.integrate.solve_ivp)
#[derive(Clone, Debug)]
pub enum Method {
    /// Bogacki–Shampine 3(2) adaptive RK
    RK23,
    /// Dormand–Prince 5(4) adaptive RK
    RK45,
    /// Dormand–Prince 8(5,3) high-order adaptive RK
    DOP853,
    /// Classic fixed-step RK4
    RK4,
}

#[derive(Builder)]
/// Options for solve_ivp similar to SciPy
pub struct IVPOptions {
    /// Method to use. Default: RK45 (Dormand–Prince 5(4)).
    #[builder(default = Method::RK45)]
    pub method: Method,
    /// Relative tolerance for error estimation.
    #[builder(default = 1e-6, into)]
    pub rtol: Tolerance,
    /// Absolute tolerance for error estimation.
    #[builder(default = 1e-6, into)]
    pub atol: Tolerance,
    /// Maximum number of allowed steps.
    pub nmax: Option<usize>,
    // Optional time points at which to store the computed solution.
    pub t_eval: Option<Vec<Float>>,
    /// Convenience alias for the initial step suggestion (maps to `settings.h0`).
    pub first_step: Option<Float>,
    /// Convenience alias for maximum step size (maps to `settings.hmax`).
    pub max_step: Option<Float>,
    /// Minimum step size constraint (maps to `settings.hmin`).
    pub min_step: Option<Float>,
    /// If true, collect dense output coefficients for per-step interpolation.
    /// When enabled, the solver returns a `dense_output` object that can
    /// evaluate the solution at arbitrary times inside the integration range.
    #[builder(default = false)]
    pub dense_output: bool,
}
