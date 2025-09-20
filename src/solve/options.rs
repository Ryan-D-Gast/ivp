//! Options for solve_ivp

use bon::Builder;

use crate::{Float, matrix::MatrixStorage, methods::Tolerance};

/// Numerical methods for solve_ivp
#[derive(Clone, Debug)]
pub enum Method {
    /// Bogacki–Shampine 3(2) adaptive RK
    RK23,
    /// Dormand–Prince 5(4) adaptive RK; in SciPy known as RK45
    DOPRI5,
    /// Dormand–Prince 8(5,3) high-order adaptive RK
    DOP853,
    /// Classic fixed-step RK4
    RK4,
    /// Radau 5th order implicit Runge-Kutta method
    RADAU,
}

impl From<&str> for Method {
    fn from(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "RK23" => Method::RK23,
            "DOPRI5" | "RK45" => Method::DOPRI5,
            "DOP853" => Method::DOP853,
            "RK4" => Method::RK4,
            "RADAU5" | "RADAU" => Method::RADAU,
            _ => Method::DOPRI5, // Default
        }
    }
}

#[derive(Builder)]
/// Options for `solve_ivp`.
pub struct Options {
    /// Integration method. Choose an explicit RK (RK23/DOPRI5/DOP853/RK4) for non‑stiff
    /// problems or an implicit RK (RADAU) for stiff/DAE systems. Default: DOPRI5 (aka RK45).
    #[builder(default = Method::DOPRI5, into)]
    pub method: Method,
    /// Relative tolerance for local error control. Accepts scalar or per‑component array/vector
    /// via [`Tolerance`]. The effective scaling for component i is `atol[i] + rtol[i]*|y[i]|`.
    #[builder(default = 1e-3, into)]
    pub rtol: Tolerance,
    /// Absolute tolerance for local error control. Accepts scalar or per‑component array/vector.
    /// Used together with `rtol` to build the error scale `atol + rtol*|y|`.
    #[builder(default = 1e-6, into)]
    pub atol: Tolerance,
    /// Maximum number of solver steps.
    pub max_steps: Option<usize>,
    /// Times at which to return the solution. If `None` the points are selected by the solver.
    pub t_eval: Option<Vec<Float>>,
    /// Initial step size (its sign must match `xend - x0`).
    pub first_step: Option<Float>,
    /// Upper bound on step size.
    pub max_step: Option<Float>,
    /// Lower bound on step size.
    pub min_step: Option<Float>,
    /// Store per‑step interpolants for cheap post‑run evaluation via `Solution::sol`/`sol_many`.
    /// Increases memory usage; recommended if you need values at times other than internal steps.
    #[builder(default = false)]
    pub dense_output: bool,
    /// Preferred storage for the Jacobian `J = ∂f/∂y`. Default: `Full` (dense, writable).
    /// Solvers that don’t use a Jacobian ignore this. For banded storage you must provide
    /// an analytical Jacobian consistent with the chosen layout.
    #[builder(default = MatrixStorage::Full)]
    pub jac_storage: MatrixStorage,
    /// Preferred storage for the mass matrix `M` in `M y' = f(t,y)`. Default: `Identity`
    /// (implicit I, no allocation). Set to `Full`/`Banded` to provide a non‑trivial mass matrix.
    #[builder(default = MatrixStorage::Identity)]
    pub mass_storage: MatrixStorage,
    /// DAE partition: number of index‑1 (differential) variables at the start of the state.
    /// If `nind2`/`nind3` are set and `nind1` is omitted, it is inferred as `n − nind2 − nind3`.
    /// If none are set, all variables are treated as index‑1 (pure ODE).
    pub nind1: Option<usize>,
    /// DAE partition: number of index‑2 algebraic variables following the index‑1 block.
    /// In RADAU error estimation these components are scaled by the current step size `h`.
    pub nind2: Option<usize>,
    /// DAE partition: number of index‑3 algebraic variables following the index‑2 block.
    /// In RADAU error estimation these components are scaled by `h^2`.
    pub nind3: Option<usize>,
}
