//! Internal fixed-step symplectic solve implementations.

use crate::{
    Float,
    dense::DenseSegment,
    error::{ConfigError, Error},
    ivp::{SecondOrderSystem, SeparableHamiltonianSystem},
    methods::{
        SymplecticMethod, SymplecticWork, drift_kick_step, kick_drift_step, ruth3_step,
        velocity_verlet_step, yoshida4_step,
    },
    solve::cont::ContinuousOutput,
    status::Status,
};

use super::solution::Solution;

#[derive(Clone, Debug)]
/// Internal fixed-step symplectic configuration.
pub(crate) struct SymplecticConfig {
    /// Symplectic method to use. Default: `VelocityVerlet`.
    pub method: SymplecticMethod,
    /// Fixed step size. Must be non-zero and its sign must match `tf - t0`.
    pub step_size: Float,
    /// Maximum number of substeps. Default: unlimited.
    pub max_steps: Option<usize>,
    /// Times at which to return the solution. If `None`, all internal step
    /// endpoints are stored.
    pub t_eval: Option<Vec<Float>>,
    /// Store per-step dense interpolants for cheap post-run evaluation.
    pub dense_output: bool,
    /// Parameters passed to the system functions.
    pub p: Vec<Float>,
}

struct SecondOrderAdapter<'a, F> {
    problem: &'a F,
    p: &'a [Float],
}

impl<'a, F> SecondOrderAdapter<'a, F> {
    fn new(problem: &'a F, p: &'a [Float]) -> Self {
        Self { problem, p }
    }
}

impl<F> SeparableHamiltonianSystem for SecondOrderAdapter<'_, F>
where
    F: SecondOrderSystem,
{
    fn position_derivative(&self, _t: Float, p_state: &[Float], _p: &[Float], dqdt: &mut [Float]) {
        dqdt.copy_from_slice(p_state);
    }

    fn momentum_derivative(&self, t: Float, q: &[Float], _p: &[Float], dpdt: &mut [Float]) {
        self.problem.acceleration(t, q, self.p, dpdt);
    }
}

/// Solve a separable Hamiltonian system using a fixed-step symplectic method.
pub(crate) fn solve_hamiltonian_impl<F>(
    f: &F,
    t0: Float,
    tf: Float,
    q0: &[Float],
    p0: &[Float],
    config: SymplecticConfig,
) -> Result<Solution, Error>
where
    F: SeparableHamiltonianSystem,
{
    validate_dimensions(q0, p0)?;
    let p_params = config.p.clone();
    solve_separable_impl(f, t0, tf, q0, p0, &p_params, config)
}

/// Solve a second-order system `q'' = a(t, q)` using a fixed-step symplectic method.
pub(crate) fn solve_second_order_impl<F>(
    f: &F,
    t0: Float,
    tf: Float,
    q0: &[Float],
    v0: &[Float],
    config: SymplecticConfig,
) -> Result<Solution, Error>
where
    F: SecondOrderSystem,
{
    validate_dimensions(q0, v0)?;
    let p_params = config.p.clone();
    let adapter = SecondOrderAdapter::new(f, &p_params);
    solve_separable_impl(&adapter, t0, tf, q0, v0, &p_params, config)
}

fn solve_separable_impl<F>(
    f: &F,
    t0: Float,
    tf: Float,
    q0: &[Float],
    p0: &[Float],
    p_params: &[Float],
    config: SymplecticConfig,
) -> Result<Solution, Error>
where
    F: SeparableHamiltonianSystem,
{
    if nearly_equal(t0, tf) {
        return Ok(constant_solution(
            t0,
            q0,
            p0,
            config.t_eval.as_deref(),
            config.dense_output,
        ));
    }

    let direction = (tf - t0).signum();
    let max_steps = config.max_steps.unwrap_or(usize::MAX);
    if max_steps == 0 {
        return Err(Error::Config(ConfigError::MustBePositive {
            parameter: "max_steps",
            value: max_steps,
        }));
    }

    if config.step_size == 0.0 || config.step_size.signum() != direction {
        return Err(Error::Config(ConfigError::InvalidStepSize {
            value: config.step_size,
            expected_sign: direction,
        }));
    }

    if let Some(ts) = config.t_eval.as_deref() {
        validate_t_eval(ts, t0, tf)?;
    }

    let n = q0.len();
    let n_states = 2 * n;
    let mut t = t0;
    let mut q = q0.to_vec();
    let mut p = p0.to_vec();
    let mut work = SymplecticWork::new(n);

    let mut t_out = Vec::new();
    let mut y_out = Vec::new();
    let mut nfev = 0usize;
    let mut nstep = 0usize;
    let mut status = Status::Success;
    let mut dense_segments = config.dense_output.then_some(Vec::<DenseSegment>::new());
    let mut start_derivative = if config.dense_output {
        Some(compute_flat_derivative(
            f, t, &q, &p, p_params, &mut work, &mut nfev,
        ))
    } else {
        None
    };

    if config.t_eval.is_none() {
        push_state(&mut t_out, &mut y_out, t, &q, &p);
        while !nearly_equal(t, tf) {
            if nstep >= max_steps {
                status = Status::NeedLargerNMax;
                break;
            }
            let h = bounded_step(t, tf, config.step_size);
            let q_old = q.clone();
            let p_old = p.clone();
            step_once(
                config.method,
                f,
                t,
                h,
                &mut q,
                &mut p,
                p_params,
                &mut work,
                &mut nfev,
            );
            let t_next = t + h;
            if let (Some(segs), Some((dq0, dp0))) =
                (dense_segments.as_mut(), start_derivative.as_ref())
            {
                let end =
                    compute_flat_derivative(f, t_next, &q, &p, p_params, &mut work, &mut nfev);
                segs.push(build_dense_segment(
                    t, h, &q_old, &p_old, dq0, dp0, &q, &p, &end.0, &end.1,
                ));
                start_derivative = Some(end);
            }
            t += h;
            nstep += 1;
            if nearly_equal(t, tf) {
                t = tf;
            }
            push_state(&mut t_out, &mut y_out, t, &q, &p);
        }
    } else {
        let outputs = config.t_eval.as_ref().unwrap();
        let mut next_output = 0usize;
        while next_output < outputs.len() || !nearly_equal(t, tf) {
            let target = if next_output < outputs.len() {
                outputs[next_output]
            } else {
                tf
            };

            while !nearly_equal(t, target) {
                if nstep >= max_steps {
                    status = Status::NeedLargerNMax;
                    break;
                }
                let h = bounded_step(t, target, config.step_size);
                let q_old = q.clone();
                let p_old = p.clone();
                step_once(
                    config.method,
                    f,
                    t,
                    h,
                    &mut q,
                    &mut p,
                    p_params,
                    &mut work,
                    &mut nfev,
                );
                let t_next = t + h;
                if let (Some(segs), Some((dq0, dp0))) =
                    (dense_segments.as_mut(), start_derivative.as_ref())
                {
                    let end =
                        compute_flat_derivative(f, t_next, &q, &p, p_params, &mut work, &mut nfev);
                    segs.push(build_dense_segment(
                        t, h, &q_old, &p_old, dq0, dp0, &q, &p, &end.0, &end.1,
                    ));
                    start_derivative = Some(end);
                }
                t += h;
                nstep += 1;
                if nearly_equal(t, target) {
                    t = target;
                }
            }

            if status != Status::Success {
                break;
            }

            if next_output < outputs.len() && nearly_equal(t, outputs[next_output]) {
                push_state(&mut t_out, &mut y_out, outputs[next_output], &q, &p);
                next_output += 1;
            } else if nearly_equal(t, tf) {
                break;
            }
        }
    }

    Ok(Solution {
        t: t_out,
        y: y_out,
        t_events: Vec::new(),
        y_events: Vec::new(),
        quad: Vec::new(),
        nfev,
        njev: 0,
        nlu: 0,
        nstep,
        naccpt: nstep,
        nrejct: 0,
        status,
        continuous_sol: dense_segments
            .map(|segs| ContinuousOutput::from_dense_segments(n_states, segs)),
    })
}

#[allow(clippy::too_many_arguments)]
fn step_once<F>(
    method: SymplecticMethod,
    f: &F,
    t: Float,
    h: Float,
    q: &mut [Float],
    p: &mut [Float],
    p_params: &[Float],
    work: &mut SymplecticWork,
    nfev: &mut usize,
) where
    F: SeparableHamiltonianSystem,
{
    match method {
        SymplecticMethod::SymplecticEulerKickDrift => {
            kick_drift_step(f, t, h, q, p, p_params, work, nfev)
        }
        SymplecticMethod::SymplecticEulerDriftKick => {
            drift_kick_step(f, t, h, q, p, p_params, work, nfev)
        }
        SymplecticMethod::VelocityVerlet => {
            velocity_verlet_step(f, t, h, q, p, p_params, work, nfev)
        }
        SymplecticMethod::Ruth3 => ruth3_step(f, t, h, q, p, p_params, work, nfev),
        SymplecticMethod::Yoshida4 => yoshida4_step(f, t, h, q, p, p_params, work, nfev),
    }
}

fn validate_dimensions(q0: &[Float], p0: &[Float]) -> Result<(), Error> {
    if q0.is_empty() {
        return Err(Error::Config(ConfigError::MustBePositive {
            parameter: "state dimension",
            value: 0,
        }));
    }

    if q0.len() != p0.len() {
        return Err(Error::Config(ConfigError::DimensionMismatch {
            parameter: "momentum dimension",
            expected: q0.len(),
            actual: p0.len(),
        }));
    }

    Ok(())
}

fn validate_t_eval(ts: &[Float], t0: Float, tf: Float) -> Result<(), Error> {
    let direction = (tf - t0).signum();
    let lo = t0.min(tf);
    let hi = t0.max(tf);

    for &t in ts {
        if t < lo || t > hi {
            return Err(Error::Config(ConfigError::OutOfRange {
                parameter: "t_eval",
                value: t,
                min: lo,
                max: hi,
            }));
        }
    }

    for window in ts.windows(2) {
        let delta = window[1] - window[0];
        if delta * direction < 0.0 {
            return Err(Error::Config(ConfigError::NonMonotonicSequence {
                parameter: "t_eval",
            }));
        }
    }

    Ok(())
}

fn bounded_step(current: Float, target: Float, base_step: Float) -> Float {
    let remaining = target - current;
    if remaining.abs() <= base_step.abs() {
        remaining
    } else {
        base_step
    }
}

fn constant_solution(
    t0: Float,
    q0: &[Float],
    p0: &[Float],
    t_eval: Option<&[Float]>,
    dense_output: bool,
) -> Solution {
    let mut t = Vec::new();
    let mut y = Vec::new();

    if let Some(ts) = t_eval {
        for &ti in ts {
            if nearly_equal(ti, t0) {
                push_state(&mut t, &mut y, ti, q0, p0);
            }
        }
    } else {
        push_state(&mut t, &mut y, t0, q0, p0);
    }

    let continuous_sol = if dense_output {
        let n_states = q0.len() + p0.len();
        let zero = vec![0.0; q0.len()];
        Some(ContinuousOutput::from_dense_segments(
            n_states,
            vec![build_dense_segment(
                t0, 1e-15, q0, p0, &zero, &zero, q0, p0, &zero, &zero,
            )],
        ))
    } else {
        None
    };

    Solution {
        t,
        y,
        t_events: Vec::new(),
        y_events: Vec::new(),
        quad: Vec::new(),
        nfev: 0,
        njev: 0,
        nlu: 0,
        nstep: 0,
        naccpt: 0,
        nrejct: 0,
        status: Status::Success,
        continuous_sol,
    }
}

fn compute_flat_derivative<F>(
    f: &F,
    t: Float,
    q: &[Float],
    p: &[Float],
    p_params: &[Float],
    work: &mut SymplecticWork,
    nfev: &mut usize,
) -> (Vec<Float>, Vec<Float>)
where
    F: SeparableHamiltonianSystem,
{
    f.position_derivative(t, p, p_params, &mut work.dqdt);
    *nfev += 1;
    f.momentum_derivative(t, q, p_params, &mut work.dpdt);
    *nfev += 1;
    (work.dqdt.clone(), work.dpdt.clone())
}

#[allow(clippy::too_many_arguments)]
fn build_dense_segment(
    xold: Float,
    h: Float,
    q0: &[Float],
    p0: &[Float],
    dq0: &[Float],
    dp0: &[Float],
    q1: &[Float],
    p1: &[Float],
    dq1: &[Float],
    dp1: &[Float],
) -> DenseSegment {
    let n = q0.len() + p0.len();
    let nq = q0.len();
    let mut cont = vec![0.0; 4 * n];

    cont[0..nq].copy_from_slice(q0);
    cont[nq..n].copy_from_slice(p0);

    cont[n..n + nq].copy_from_slice(dq0);
    cont[n + nq..2 * n].copy_from_slice(dp0);

    cont[2 * n..2 * n + nq].copy_from_slice(q1);
    cont[2 * n + nq..3 * n].copy_from_slice(p1);

    cont[3 * n..3 * n + nq].copy_from_slice(dq1);
    cont[3 * n + nq..4 * n].copy_from_slice(dp1);

    DenseSegment::new(cont, xold, h, interpolate_hermite)
}

fn interpolate_hermite(xi: Float, yi: &mut [Float], cont: &[Float], xold: Float, h: Float) {
    let t = (xi - xold) / h;
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    let n = yi.len();
    for i in 0..n {
        yi[i] = h00 * cont[i]
            + h10 * h * cont[n + i]
            + h01 * cont[2 * n + i]
            + h11 * h * cont[3 * n + i];
    }
}

fn push_state(
    t_out: &mut Vec<Float>,
    y_out: &mut Vec<Vec<Float>>,
    t: Float,
    q: &[Float],
    p: &[Float],
) {
    t_out.push(t);
    let mut y = Vec::with_capacity(q.len() + p.len());
    y.extend_from_slice(q);
    y.extend_from_slice(p);
    y_out.push(y);
}

fn nearly_equal(a: Float, b: Float) -> bool {
    let scale = a.abs().max(b.abs()).max(1.0);
    (a - b).abs() <= 16.0 * Float::EPSILON * scale
}
