//! Solve an initial value problem for a system of ODEs.

use crate::{
    Float,
    core::ode::ODE,
    error::Error,
    methods::{
        dp::{dop853, dopri5},
        rk::{rk4, rk23},
        settings::Settings,
    },
};

use super::{
    cont::ContinuousOutput,
    options::{IVPOptions, Method},
    solout::DefaultSolOut,
    solution::IVPSolution,
};

/// Solve an initial value problem (IVP) for a system of first‑order ODEs: y' = f(x, y).
///
/// This integrates from `x0` to `xend` starting at state `y0` using the method and
/// tolerances specified in `options`. The result contains the discrete samples, solver
/// statistics, and (optionally) a continuous interpolant for dense evaluation.
///
/// Arguments:
/// - `f`: System right‑hand side implementing `core::ode::ODE`.
/// - `x0`: Initial independent variable (time) value.
/// - `xend`: Final independent variable value. Can be less than `x0` to integrate backward.
/// - `y0`: Initial state vector at `x0`.
/// - `options`: Integration options (method, rtol/atol, step size limits, `t_eval`,
///   and whether to build dense output).
///
/// Returns:
/// - `Ok(IVPSolution)`: Discrete samples and stats. Fields:
///   - `t`: Sampled time points (either internal accepted steps or `options.t_eval` if provided)
///   - `y`: State values corresponding to `t` (shape: `t.len() x y0.len()`)
///   - `nfev`, `nstep`, `naccpt`, `nrejct`, `status`: Solver statistics and final status
///   - Continuous evaluation is available via `IVPSolution::sol`, `sol_many`, `sol_span`
///     when `options.dense_output == true`.
/// - `Err(Vec<Error>)`: One or more errors encountered during integration.
///
/// Notes:
/// - Sampling:
///   - If `options.t_eval` is `Some(ts)`, the solver reports exactly those times in `t`
///     (subject to solver success).
///   - Otherwise, `t` and `y` contain all accepted internal steps.
/// - Dense output:
///   - If enabled via `options.dense_output`, the returned `IVPSolution` exposes
///     `sol(t)` and `sol_many(&ts)` for continuous interpolation inside the covered span.
///   - `sol_many` always returns `Vec<Option<Vec<Float>>>`; if dense output is disabled,
///     it yields a vector of `None` values of the same length as `ts`.
/// - Direction:
///   - The solver infers the integration direction from `xend - x0` and handles forward
///     and backward integration.
///
/// Example:
/// ```rust,no_run
/// use ivp::prelude::*;
///
/// struct SHO;
/// impl ODE for SHO {
///     fn ode(&self, _x: f64, y: &[f64], dydx: &mut [f64]) {
///         dydx[0] = y[1];
///         dydx[1] = -y[0];
///     }
/// }
///
/// fn main() {
///     let opts = IVPOptions::builder()
///         .rtol(1e-9).atol(1e-9)
///         .method(Method::DOP853)
///         .dense_output(true)
///         .build();
///
///     let f = SHO;
///     let x0 = 0.0;
///     let xend = 2.0 * std::f64::consts::PI; // one period
///     let y0 = [1.0, 0.0];
///
///     let sol = solve_ivp(&f, x0, xend, &y0, opts).unwrap();
///
///     // Iterate over stored samples
///     for (t, y) in sol.iter() {
///         // ...
///     }
///
///     // Evaluate continuous solution (if enabled)
///     if let Some((t0, t1)) = sol.sol_span() {
///         let mid = 0.5 * (t0 + t1);
///         let y_mid = sol.sol(mid); // Option<Vec<f64>>
///         let ys = sol.sol_many(&[t0, mid, t1]); // Vec<Option<Vec<f64>>>
///     }
/// }
/// ```
pub fn solve_ivp<F>(
    f: &F,
    x0: Float,
    xend: Float,
    y0: &[Float],
    options: IVPOptions,
) -> Result<IVPSolution, Vec<Error>>
where
    F: ODE,
{
    // Build Settings (rtol/atol are passed to methods)
    let settings = Settings::builder()
        .maybe_nmax(options.nmax)
        .maybe_h0(options.first_step)
        .maybe_hmax(options.max_step)
        .maybe_hmin(options.min_step)
        .build();

    // Prepare the default SolOut (wrapping user callback if provided)
    let mut default_solout = DefaultSolOut::new(options.t_eval, options.dense_output);

    // Dispatch by method
    let result = match options.method {
        Method::RK4 => {
            let h = settings.h0.unwrap_or_else(|| (xend - x0) / 100.0);
            rk4(f, x0, xend, y0, h, Some(&mut default_solout), settings)
        }
        Method::RK23 => rk23(
            f,
            x0,
            xend,
            y0,
            options.rtol,
            options.atol,
            Some(&mut default_solout),
            settings,
        ),
        Method::DOPRI5 => dopri5(
            f,
            x0,
            xend,
            y0,
            options.rtol,
            options.atol,
            Some(&mut default_solout),
            settings,
        ),
        Method::DOP853 => dop853(
            f,
            x0,
            xend,
            y0,
            options.rtol,
            options.atol,
            Some(&mut default_solout),
            settings,
        ),
    };

    match result {
        Ok(sol) => {
            let (t, y, dense_raw) = default_solout.into_payload();
            let continuous_sol = if options.dense_output {
                Some(ContinuousOutput::from_segments(options.method, dense_raw))
            } else {
                None
            };
            Ok(IVPSolution {
                t,
                y,
                nfev: sol.nfev,
                nstep: sol.nstep,
                naccpt: sol.naccpt,
                nrejct: sol.nrejct,
                status: sol.status,
                continuous_sol,
            })
        }
        Err(errors) => {
            return Err(errors);
        }
    }
}
