//! A SciPy-like entry point to solve an IVP for ODEs with convenient options.

use bon::Builder;

use crate::Float;
use crate::{
    error::Error,
    methods::{
        dp::{dop853, dopri5},
        rk::{rk4, rk23},
        settings::Settings,
    },
    prelude::{Interpolate, ODE, SolOut, Status},
};

#[derive(Builder)]
/// Options for solve_ivp similar to SciPy
pub struct IVPOptions<'a, S: SolOut> {
    /// Method to use. Default: RK45 (Dormand–Prince 5(4)).
    #[builder(default = Method::RK45)]
    pub method: Method,
    /// Relative tolerance for error estimation.
    #[builder(default = 1e-6)]
    pub rtol: Float,
    /// Absolute tolerance for error estimation.
    #[builder(default = 1e-6)]
    pub atol: Float,
    /// Maximum number of allowed steps.
    #[builder(default = 100_000)]
    pub nmax: usize,
    /// Points where the solution is requested. If provided, the default SolOut will
    /// use dense output to sample at these locations.
    pub t_eval: Option<Vec<Float>>,
    /// Optional user callback invoked after each accepted step.
    /// If provided together with `t_eval`, the callback will be invoked after
    /// internal sampling at `t_eval`.
    pub solout: Option<&'a mut S>,
    /// Convenience alias for the initial step suggestion (maps to `settings.h0`).
    pub first_step: Option<Float>,
    /// Convenience alias for maximum step size (maps to `settings.hmax`).
    pub max_step: Option<Float>,
    /// Minimum step size constraint (not yet enforced by methods; placeholder).
    pub min_step: Option<Float>,
    /// Save step endpoints (initial call and each accepted step). Default: true.
    #[builder(default = true)]
    pub save_step_endpoints: bool,
}

/// Solve an initial value problem with SciPy-like options.
pub fn solve_ivp<F, S>(
    f: &F,
    x0: Float,
    xend: Float,
    y0: &[Float],
    mut options: IVPOptions<'_, S>,
) -> Result<IVPSolution, Error>
where
    F: ODE,
    S: SolOut,
{
    // Build Settings from IVPOptions knobs
    let mut settings = Settings::builder()
        .rtol(options.rtol)
        .atol(options.atol)
        .nmax(options.nmax)
        .build();
    // Align convenience aliases into Settings
    settings.h0 = options.first_step;
    settings.hmax = options.max_step;

    // Validate t_eval if provided
    if let Some(ref te) = options.t_eval {
        if te.is_empty() {
            return Err(Error::InvalidTEval(
                "t_eval must be non-empty when provided".into(),
            ));
        }
        // Check monotonicity matching direction
        let dir = (xend - x0).signum();
        let mut prev = te[0];
        for &t in te.iter().skip(1) {
            if !(dir >= 0.0 && t >= prev || dir <= 0.0 && t <= prev) {
                return Err(Error::InvalidTEval(
                    "t_eval must be monotonic in the integration direction".into(),
                ));
            }
            prev = t;
        }
        // Check range coverage
        let min_t = te.first().copied().unwrap();
        let max_t = te.last().copied().unwrap();
        let (lo, hi) = if dir >= 0.0 { (x0, xend) } else { (xend, x0) };
        if min_t < lo - 1e-12 || max_t > hi + 1e-12 {
            return Err(Error::InvalidTEval(
                "t_eval points must lie within [x0, xend]".into(),
            ));
        }
    }

    // Prepare the default SolOut (wrapping user callback if provided)
    let mut default_solout = DefaultSolOut::new(
        options.t_eval.as_deref(),
        options.save_step_endpoints,
        options.solout.as_deref_mut(),
    );

    // Dispatch by method
    match options.method {
        Method::RK4 => {
            // Use settings.h0 if provided; otherwise default to (xend - x0)/100
            let h = settings.h0.unwrap_or_else(|| (xend - x0) / 100.0);
            let res = rk4(f, x0, xend, y0, h, Some(&mut default_solout), settings);
            res.map(|solution| {
                let (t, y) = default_solout.into_data();
                IVPSolution {
                    t,
                    y,
                    nfev: solution.nfev,
                    nstep: solution.nstep,
                    naccpt: solution.naccpt,
                    nrejct: solution.nrejct,
                    status: solution.status,
                }
            })
        }
        Method::RK23 => rk23(f, x0, xend, y0, Some(&mut default_solout), settings)
            .map_err(Error::MethodErrors)
            .map(|solution| {
                let (t, y) = default_solout.into_data();
                IVPSolution {
                    t,
                    y,
                    nfev: solution.nfev,
                    nstep: solution.nstep,
                    naccpt: solution.naccpt,
                    nrejct: solution.nrejct,
                    status: solution.status,
                }
            }),
        Method::RK45 => dopri5(f, x0, xend, y0, Some(&mut default_solout), settings)
            .map_err(Error::MethodErrors)
            .map(|solution| {
                let (t, y) = default_solout.into_data();
                IVPSolution {
                    t,
                    y,
                    nfev: solution.nfev,
                    nstep: solution.nstep,
                    naccpt: solution.naccpt,
                    nrejct: solution.nrejct,
                    status: solution.status,
                }
            }),
        Method::DOP853 => dop853(f, x0, xend, y0, Some(&mut default_solout), settings)
            .map_err(Error::MethodErrors)
            .map(|solution| {
                let (t, y) = default_solout.into_data();
                IVPSolution {
                    t,
                    y,
                    nfev: solution.nfev,
                    nstep: solution.nstep,
                    naccpt: solution.naccpt,
                    nrejct: solution.nrejct,
                    status: solution.status,
                }
            }),
    }
}

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

/// Rich solution of solve_ivp: the base integrator `Solution` plus sampled data from SolOut.
#[derive(Debug, Clone)]
pub struct IVPSolution {
    pub t: Vec<Float>,
    pub y: Vec<Vec<Float>>,
    pub nfev: usize,
    pub nstep: usize,
    pub naccpt: usize,
    pub nrejct: usize,
    pub status: Status,
}

/// Default SolOut that implements t_eval sampling and endpoint recording; wraps a user SolOut.
struct DefaultSolOut<'a, S: SolOut> {
    t_eval: Option<&'a [Float]>,
    save_endpoints: bool,
    next_idx: usize,
    tol: Float,
    t: Vec<Float>,
    y: Vec<Vec<Float>>, // y[i] corresponds to t[i]
    user: Option<&'a mut S>,
}

impl<'a, S: SolOut> DefaultSolOut<'a, S> {
    fn new(t_eval: Option<&'a [Float]>, save_endpoints: bool, user: Option<&'a mut S>) -> Self {
        Self {
            t_eval,
            save_endpoints,
            next_idx: 0,
            tol: 1e-12,
            t: Vec::new(),
            y: Vec::new(),
            user,
        }
    }

    fn into_data(self) -> (Vec<Float>, Vec<Vec<Float>>) {
        (self.t, self.y)
    }
}

impl<'a, S: SolOut> SolOut for DefaultSolOut<'a, S> {
    fn solout<I: Interpolate>(
        &mut self,
        xold: Float,
        x: Float,
        y: &[Float],
        interpolator: &I,
    ) -> crate::prelude::ControlFlag {
        // Record endpoints
        if self.save_endpoints {
            if (xold - x).abs() <= self.tol {
                self.t.push(x);
                self.y.push(y.to_vec());
            } else {
                self.t.push(x);
                self.y.push(y.to_vec());
            }
        }

        // If t_eval is provided, interpolate and store values within (xold, x]
        if let Some(te) = self.t_eval {
            // Handle the initial call (xold == x) -> only include exact match
            let mut i = self.next_idx;
            if (xold - x).abs() <= self.tol {
                while i < te.len() && (te[i] - x).abs() <= self.tol {
                    let mut yi = vec![0.0; y.len()];
                    interpolator.interpolate(te[i], &mut yi);
                    self.t.push(te[i]);
                    self.y.push(yi);
                    i += 1;
                }
            } else {
                // Regular accepted step
                // Include all te[i] in (xold, x] up to tolerance
                while i < te.len() && te[i] <= x + self.tol {
                    if te[i] >= xold - self.tol {
                        let mut yi = vec![0.0; y.len()];
                        interpolator.interpolate(te[i], &mut yi);
                        self.t.push(te[i]);
                        self.y.push(yi);
                    }
                    i += 1;
                }
            }
            self.next_idx = i;
        }

        // Forward to user callback if any
        if let Some(user) = self.user.as_deref_mut() {
            return user.solout(xold, x, y, interpolator);
        }

        crate::prelude::ControlFlag::Continue
    }
}
