use crate::{Float, ODE, SolOut, ControlFlag, Tolerance};
use crate::rk::{RKSettings, RKResult};
use crate::status::Status;
use crate::error::Error;
use crate::interpolate::Interpolate;

/// Bogacki–Shampine 3(2) pair (RK23) adaptive-step integrator.
/// This implementation uses an embedded method to estimate errors
/// and adjust the step size accordingly with dense output.
pub fn rk23<'a, F, S, R, A>(
    f: &F,
    x: Float,
    y: &[Float],
    xend: Float,
    rtol: R,
    atol: A,
    solout: &mut S,
    settings: RKSettings,
) -> Result<RKResult, Error>
where
    F: ODE,
    S: SolOut,
    R: Into<Tolerance<'a>>,
    A: Into<Tolerance<'a>>,
{

    let rtol = rtol.into();
    let atol = atol.into();

    let result = rk23_core(f, x, y.to_vec(), xend, rtol, atol, solout, settings);

    Ok(result)
}

fn rk23_core<'a, F, S>(
    f: &F,
    mut x: Float,
    mut y: Vec<Float>,
    xend: Float,
    rtol: Tolerance<'a>,
    atol: Tolerance<'a>,
    solout: &mut S,
    settings: RKSettings,
) -> RKResult
where
    F: ODE,
    S: SolOut,
{
    let n = y.len();
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut yt = vec![0.0; n];
    let mut y_err = vec![0.0; n];
    let mut nfev = 0;
    let mut nstep = 0;
    let mut status = Status::Success;
    let mut xold;
    let mut h = settings.h0.unwrap_or((xend - x).abs() / 10.0);
    let direction = (xend - x).signum();

    // Initial derivative
    f.ode(x, &y, &mut k1);
    nfev += 1;

    loop {
        if nstep >= settings.nmax {
            status = Status::NeedLargerNmax;
            break;
        }

        if (x + h - xend) * direction > 0.0 {
            h = xend - x;
        }

        // Stage 2
        for i in 0..n {
            yt[i] = y[i] + h * A21 * k1[i];
        }
        f.ode(x + C2 * h, &yt, &mut k2);
        nfev += 1;

        // Stage 3
        for i in 0..n {
            yt[i] = y[i] + h * A32 * k2[i];
        }
        f.ode(x + C3 * h, &yt, &mut k3);
        nfev += 1;

        // Compute solution and error estimate
        for i in 0..n {
            yt[i] = y[i] + h * (B1 * k1[i] + B2 * k2[i] + B3 * k3[i]);
        }

        // Stage 4-1: derivative at new point, also used as k1 if accepted.
        f.ode(x + h, &yt, &mut k4);
        nfev += 1;

        // Error estimate using embedded 2nd order solution
        for i in 0..n {
            y_err[i] = h * (E1 * k1[i] + E2 * k2[i] + E3 * k3[i] + E4 * k4[i]);
        }

        // Error estimation
        let mut err = 0.0;
        for i in 0..n {
            let tol = atol[i] + rtol[i] * yt[i].abs().max(y[i].abs());
            err += (y_err[i] / tol).powi(2);
        }
        err = (err / n as Float).sqrt();

        if err <= 1.0 {
            // Step accepted
            xold = x;
            x += h;
            y.copy_from_slice(&yt);
            nstep += 1;

            let dense_output = dense_output_impl(xold, x, &y, &k1, &k2, &k3, &k4);

            let flag = solout.solout(x - h, x, &y, &dense_output);
            match flag {
                ControlFlag::Interrupt => {
                    status = Status::Interrupted;
                    break;
                }
                ControlFlag::ModifiedSolution => {
                    // User changed y in-place; recompute k1 at new (x, y).
                    f.ode(x, &y, &mut k1);
                    nfev += 1;
                }
                ControlFlag::Continue => {
                    // Reuse k4 as k1 for the next step to save an evaluation.
                    k1.copy_from_slice(&k4);
                }
            }

            if x == xend {
                break;
            }

            // Adjust step size
            h *= (0.9 * err.powf(-1.0 / 3.0)).min(5.0).max(0.2);
        } else {
            // Step rejected
            h *= (0.9 * err.powf(-1.0 / 3.0)).min(1.0).max(0.2);
        }
    }

    RKResult {
        x,
        y,
        h,
        nfev,
        nstep,
        status,
    }
}

// Coefficients for RK23
const C2: Float = 0.5;
const C3: Float = 0.75;
const A21: Float = 0.5;
const A32: Float = 0.75;
const B1: Float = 2.0 / 9.0;
const B2: Float = 1.0 / 3.0;
const B3: Float = 4.0 / 9.0;
const E1: Float = 5.0 / 72.0;
const E2: Float = -1.0 / 12.0;
const E3: Float = -1.0 / 9.0;
const E4: Float = 1.0 / 8.0;

pub struct RkDenseOutput<'a> {
    t_old: Float,
    y_old: &'a [Float],
    q: Vec<Vec<Float>>,
    h: Float,
}

impl<'a> RkDenseOutput<'a> {
    pub fn new(t_old: Float, t: Float, y_old: &'a [Float], q: Vec<Vec<Float>>) -> Self {
        let h = t - t_old;
        Self { t_old, y_old, q, h }
    }
}

impl<'a> Interpolate for RkDenseOutput<'a> {
    fn interpolate(&self, t: Float, y: &mut [Float]) {
        let x = (t - self.t_old) / self.h;
        // Basis is [x, x^2, x^3]
        let mut p = vec![0.0; self.q[0].len()];
        if !p.is_empty() {
            p[0] = x;
            for i in 1..p.len() {
                p[i] = p[i - 1] * x;
            }
        }

        for (i, row) in self.q.iter().enumerate() {
            let mut yi = self.y_old[i];
            for (j, &q_ij) in row.iter().enumerate() {
                yi += self.h * q_ij * p[j];
            }
            y[i] = yi;
        }
    }
}

fn dense_output_impl<'a>(
    t_old: Float,
    t: Float,
    y_old: &'a [Float],
    k1: &[Float],
    k2: &[Float],
    k3: &[Float],
    k4: &[Float],
) -> RkDenseOutput<'a> {
    // Build Q coefficients directly:
    // Q0 = k1
    // Q1 = -4/3*k1 + 1*k2 + 4/3*k3 - 1*k4
    // Q2 =  5/9*k1 - 2/3*k2 - 8/9*k3 + 1*k4
    let n = k1.len();
    let mut q = vec![vec![0.0; 3]; n];
    for i in 0..n {
        let q0 = k1[i];
        let q1 = (-4.0 / 3.0) * k1[i] + k2[i] + (4.0 / 3.0) * k3[i] - k4[i];
        let q2 = (5.0 / 9.0) * k1[i] - (2.0 / 3.0) * k2[i] - (8.0 / 9.0) * k3[i] + k4[i];
        q[i][0] = q0;
        q[i][1] = q1;
        q[i][2] = q2;
    }
    RkDenseOutput::new(t_old, t, y_old, q)
}
