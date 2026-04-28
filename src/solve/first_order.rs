//! Internal first-order solve implementation.

use crate::{
    Float,
    error::Error,
    ivp::FirstOrderSystem,
    methods::{BDF, DOP853, DOPRI5, LSODA, RADAU, RK4, RK23},
};

use super::{
    cont::ContinuousOutput,
    options::{FirstOrderConfig, Method},
    solout::DefaultSolOut,
    solution::Solution,
};

pub(crate) fn solve_first_order_impl<F>(
    f: &F,
    x0: Float,
    xend: Float,
    y0: &[Float],
    mut config: FirstOrderConfig,
) -> Result<Solution, Error>
where
    F: FirstOrderSystem,
{
    let p = &mut config.p;

    // Handle zero-interval case: when x0 == xend, return immediately with initial state
    if (xend - x0).abs() < 1e-15 {
        // If t_eval is provided, return all t_eval points that match x0
        let (t, y) = if let Some(ref t_eval) = config.t_eval {
            let matching: Vec<_> = t_eval
                .iter()
                .filter(|&&t| (t - x0).abs() < 1e-12)
                .copied()
                .collect();
            let y_vals: Vec<Vec<Float>> = matching.iter().map(|_| y0.to_vec()).collect();
            (matching, y_vals)
        } else {
            (vec![x0], vec![y0.to_vec()])
        };

        // Create a "constant" ContinuousOutput if dense_output is requested
        // This allows sol(t) to return y0 for any t (with extrapolation)
        let continuous_sol = if config.dense_output {
            Some(ContinuousOutput::constant(config.method, x0, y0))
        } else {
            None
        };

        return Ok(Solution {
            t,
            y,
            t_events: vec![Vec::new(); f.n_events()],
            y_events: vec![Vec::new(); f.n_events()],
            quad: vec![0.0; f.n_quadrature()],
            nfev: 0,
            njev: 0,
            nlu: 0,
            nstep: 0,
            naccpt: 0,
            nrejct: 0,
            status: crate::status::Status::Success,
            continuous_sol,
        });
    }

    // Handle empty state vector case: nothing to integrate
    if y0.is_empty() {
        let t = if let Some(ref t_eval) = config.t_eval {
            t_eval.clone()
        } else {
            vec![x0, xend]
        };
        let y: Vec<Vec<Float>> = t.iter().map(|_| Vec::new()).collect();

        let continuous_sol = if config.dense_output {
            Some(ContinuousOutput::constant(config.method, x0, y0))
        } else {
            None
        };

        return Ok(Solution {
            t,
            y,
            t_events: vec![Vec::new(); f.n_events()],
            y_events: vec![Vec::new(); f.n_events()],
            quad: vec![0.0; f.n_quadrature()],
            nfev: 0,
            njev: 0,
            nlu: 0,
            nstep: 0,
            naccpt: 0,
            nrejct: 0,
            status: crate::status::Status::Success,
            continuous_sol,
        });
    }

    // Prepare the default SolOut (wrapping user callback if provided)
    let n_states = y0.len();
    let mut default_solout = DefaultSolOut::new(
        f,
        config.t_eval.clone(),
        config.dense_output,
        config.first_step,
        x0,
        n_states,
    );

    // Dispatch by method
    let result = match config.method {
        Method::RK4 => {
            let h = config.first_step.unwrap_or_else(|| (xend - x0) / 100.0);
            let solver = RK4::builder()
                .max_steps(config.max_steps.unwrap_or(usize::MAX))
                .build();
            solver.solve(f, x0, y0, p, xend, h, Some(&mut default_solout))
        }
        Method::RK23 => {
            let solver = RK23::builder()
                .maybe_max_step(config.max_step)
                .maybe_first_step(config.first_step)
                .max_steps(config.max_steps.unwrap_or(usize::MAX))
                .build();
            solver.solve(
                f,
                x0,
                y0,
                p,
                xend,
                config.rtol,
                config.atol,
                Some(&mut default_solout),
            )
        }
        Method::DOPRI5 => {
            let solver = DOPRI5::builder()
                .maybe_max_step(config.max_step)
                .maybe_first_step(config.first_step)
                .max_steps(config.max_steps.unwrap_or(usize::MAX))
                .build();
            solver.solve(
                f,
                x0,
                y0,
                p,
                xend,
                config.rtol,
                config.atol,
                Some(&mut default_solout),
            )
        }
        Method::DOP853 => {
            let solver = DOP853::builder()
                .maybe_max_step(config.max_step)
                .maybe_first_step(config.first_step)
                .max_steps(config.max_steps.unwrap_or(usize::MAX))
                .build();
            solver.solve(
                f,
                x0,
                y0,
                p,
                xend,
                config.rtol,
                config.atol,
                Some(&mut default_solout),
            )
        }
        Method::RADAU => {
            let solver = RADAU::builder()
                .maybe_max_step(config.max_step)
                .maybe_min_step(config.min_step)
                .maybe_first_step(config.first_step)
                .max_steps(config.max_steps.unwrap_or(usize::MAX))
                .maybe_nind1(config.nind1)
                .maybe_nind2(config.nind2)
                .maybe_nind3(config.nind3)
                .jac_storage(config.jac_storage)
                .mass_storage(config.mass_storage)
                .build();
            solver.solve(
                f,
                x0,
                y0,
                p,
                xend,
                config.rtol,
                config.atol,
                Some(&mut default_solout),
            )
        }
        Method::BDF => {
            let solver = BDF::builder()
                .maybe_max_step(config.max_step)
                .maybe_min_step(config.min_step)
                .maybe_first_step(config.first_step)
                .max_steps(config.max_steps.unwrap_or(usize::MAX))
                .jac_storage(config.jac_storage)
                .build();
            solver.solve(
                f,
                x0,
                y0,
                p,
                xend,
                config.rtol,
                config.atol,
                Some(&mut default_solout),
            )
        }
        Method::LSODA => {
            let solver = LSODA::builder()
                .max_steps(config.max_steps.unwrap_or(usize::MAX))
                .maybe_max_step(config.max_step)
                .maybe_min_step(config.min_step)
                .maybe_first_step(config.first_step)
                .jac_storage(config.jac_storage)
                .jacobian_source(config.jacobian_source)
                .build();
            solver.solve(
                f,
                x0,
                y0,
                p,
                xend,
                config.rtol,
                config.atol,
                Some(&mut default_solout),
            )
        }
    };

    match result {
        Ok(sol) => {
            let (t, y, t_events, y_events, dense_raw, quad) = default_solout.into_payload();
            let continuous_sol = if config.dense_output {
                Some(ContinuousOutput::from_segments(
                    config.method,
                    n_states,
                    dense_raw,
                ))
            } else {
                None
            };
            Ok(Solution {
                t,
                y,
                t_events,
                y_events,
                quad,
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
