//! Options and method selection for solve_ivp

use bon::Builder;

use crate::{
    Float,
    methods::settings::Tolerance,
    prelude::{SolOut},
};

use i_v_p_options_builder::{IsUnset, State, SetTEval, SetSaveStepEndpoints};

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
pub struct IVPOptions<'a, S: SolOut> {
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
    /// Points where the solution is requested. If provided, the default SolOut will
    /// use dense output to sample at these locations. By default this will disable
    /// [`save_step_endpoints`](Self::save_step_endpoints) unless explicitly set.
    #[builder(setters(vis = "", name = t_eval_internal))]
    pub t_eval: Option<Vec<Float>>,
    /// Optional user callback invoked after each accepted step.
    /// If provided together with `t_eval`, the callback will be invoked after
    /// internal sampling at `t_eval`.
    pub solout: Option<&'a mut S>,
    /// Convenience alias for the initial step suggestion (maps to `settings.h0`).
    pub first_step: Option<Float>,
    /// Convenience alias for maximum step size (maps to `settings.hmax`).
    pub max_step: Option<Float>,
    /// Minimum step size constraint (maps to `settings.hmin`).
    pub min_step: Option<Float>,
    /// Save step endpoints (initial call and each accepted step). Default: true.
    #[builder(default = true)]
    pub save_step_endpoints: bool,
}

impl<'a, SOLOUT: SolOut, STATE: State> IVPOptionsBuilder<'a, SOLOUT, STATE> {
    pub fn t_eval(self, t_eval: Vec<Float>) -> IVPOptionsBuilder<'a, SOLOUT, SetTEval<SetSaveStepEndpoints<STATE>>> 
    where 
        STATE::TEval: IsUnset,
        STATE::SaveStepEndpoints: IsUnset,
    {
        self.save_step_endpoints(false).t_eval_internal(t_eval)
    }
}