//! Solve an initial value problem for a system of ODEs.

use crate::{
    error::Error,
    methods::{BDF, DOP853, DOPRI5, RADAU, RK23, RK4},
    ivp::IVP,
    Float,
};

use super::{
    cont::ContinuousOutput,
    options::{Method, Options},
    solout::DefaultSolOut,
    solution::Solution,
};

/// Solve an initial value problem (IVP) for a system of first‑order ODEs: y' = f(x, y).
///
/// This integrates from `x0` to `xend` starting at state `y0` using the method and
/// tolerances specified in `options`. The result contains the discrete samples, solver
/// statistics, and (optionally) a continuous interpolant for dense evaluation.
///
/// Arguments:
/// - `f`: System right‑hand side implementing `IVP`.
/// - `x0`: Initial independent variable (time) value.
/// - `xend`: Final independent variable value. Can be less than `x0` to integrate backward.
/// - `y0`: Initial state vector at `x0`.
/// - `options`: Integration options (method, rtol/atol, step size limits, `t_eval`,
///   and whether to build dense output).
///
/// Returns:
/// - `Ok(Solution)`: Discrete samples and stats. Fields:
///   - `t`: Sampled time points (either internal accepted steps or `options.t_eval` if provided)
///   - `y`: State values corresponding to `t` (shape: `t.len() x y0.len()`)
///   - `nfev`, `nstep`, `naccpt`, `nrejct`, `status`: Solver statistics and final status
///   - Continuous evaluation is available via `Solution::sol`, `sol_many`, `sol_span`
///     when `options.dense_output == true`.
/// - `Err(Vec<Error>)`: One or more errors encountered during integration.
///
/// Notes:
/// - Sampling:
///   - If `options.t_eval` is `Some(ts)`, the solver reports exactly those times in `t`
///     (subject to solver success).
///   - Otherwise, `t` and `y` contain all accepted internal steps.
/// - Dense output:
///   - If enabled via `options.dense_output`, the returned `Solution` exposes
///     `sol(t) -> Result<Vec<Float>, Error>` and `sol_many(&ts) -> Result<Vec<Vec<Float>>, Error>`
///     for continuous interpolation inside the covered span.
///   - Both functions return an `Err` if dense output was not requested or if any
///     requested time lies outside the covered interval. `sol_many` returns a
///     single `Err` in that case. This can easily be checked via `sol.sol_span()`.
/// - Direction:
///   - The solver infers the integration direction from `xend - x0` and handles forward
///     and backward integration.
///
/// Example:
/// ```
/// use ivp::prelude::*;
///
/// struct SHO;
/// impl IVP for SHO {
///     fn ode(&self, _x: f64, y: &[f64], dydx: &mut [f64]) {
///         dydx[0] = y[1];
///         dydx[1] = -y[0];
///     }
/// }
///
/// fn main() {
///     let opts = Options::builder()
///         .method(Method::DOP853)
///         .rtol(1e-9).atol(1e-9)
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
///     // Discrete samples
///     println!("Discrete output at accepted steps:");
///     for (t, y) in sol.iter() {
///         println!("x = {:>8.5}, y = {:?}", t, y);
///     }
///
///     // Continuous evaluation within the solution span
///     if let Some((t0, t1)) = sol.sol_span() {
///         let ts = [t0, 0.5*(t0+t1), t1];
///         let ys = sol.sol_many(&ts).unwrap();
///         println!("\nDense output at t0, (t0+t1)/2, t1:");
///         for (t, y) in ts.iter().zip(ys.iter()) {
///             println!("x = {:>8.5}, y = {:?}", t, y);
///         }
///     }
/// }
/// ```
pub fn solve_ivp<F>(
    f: &F,
    x0: Float,
    xend: Float,
    y0: &[Float],
    options: Options,
) -> Result<Solution, Vec<Error>>
where
    F: IVP,
{
    // Prepare the default SolOut (wrapping user callback if provided)
    let mut default_solout = DefaultSolOut::new(f, options.t_eval, options.dense_output, options.first_step, x0);

    // Dispatch by method
    let result = match options.method {
        Method::RK4 => {
            let h = options.first_step.unwrap_or_else(|| (xend - x0) / 100.0);
            let mut y = y0.to_vec();
            RK4::solve(
                f,
                x0,
                xend,
                &mut y,
                h,
                Some(&mut default_solout),
                true,
                options.max_steps,
            )
        }
        Method::RK23 => {
            let mut y = y0.to_vec();
            RK23::solve(
                f,
                x0,
                xend,
                &mut y,
                options.rtol,
                options.atol,
                Some(&mut default_solout),
                true,
                None,
                None,
                None,
                options.max_step,
                options.first_step,
                options.max_steps,
            )
        }
        Method::DOPRI5 => {
            let mut y = y0.to_vec();
            DOPRI5::solve(
                f,
                x0,
                xend,
                &mut y,
                options.rtol,
                options.atol,
                Some(&mut default_solout),
                true,
                None,
                None,
                None,
                None,
                None,
                options.max_step,
                options.first_step,
                options.max_steps,
                None,
            )
        }
        Method::DOP853 => {
            let mut y = y0.to_vec();
            DOP853::solve(
                f,
                x0,
                xend,
                &mut y,
                options.rtol,
                options.atol,
                Some(&mut default_solout),
                true,
                None,
                None,
                None,
                None,
                None,
                options.max_step,
                options.first_step,
                options.max_steps,
                None,
            )
        }
        Method::RADAU => {
            let mut y = y0.to_vec();
            RADAU::solve(
                f,
                x0,
                xend,
                &mut y,
                options.rtol,
                options.atol,
                Some(&mut default_solout),
                true,
                options.max_steps,
                None,
                None,
                None,
                None,
                options.max_step,
                options.min_step,
                None,
                None,
                None,
                options.nind1,
                options.nind2,
                options.nind3,
                Some(options.jac_storage),
                Some(options.mass_storage),
                options.first_step,
            )
        }
        Method::BDF => {
            let mut y = y0.to_vec();
            BDF::solve(
                f,
                x0,
                xend,
                &mut y,
                options.rtol,
                options.atol,
                Some(&mut default_solout),
                options.max_steps,
                options.max_step,
                options.min_step,
                None,
                None,
                Some(options.jac_storage),
                options.first_step,
            )
        }
    };

    match result {
        Ok(sol) => {
            let (t, y, t_events, y_events, dense_raw) = default_solout.into_payload();
            let continuous_sol = if options.dense_output {
                Some(ContinuousOutput::from_segments(options.method, dense_raw))
            } else {
                None
            };
            Ok(Solution {
                t,
                y,
                t_events,
                y_events,
                nfev: sol.evals.ode,
                njev: sol.evals.jac,
                nlu: sol.evals.lu,
                nstep: sol.steps.total,
                naccpt: sol.steps.accepted,
                nrejct: sol.steps.rejected,
                status: sol.status,
                continuous_sol,
            })
        }
        Err(errors) => Err(errors),
    }
}
