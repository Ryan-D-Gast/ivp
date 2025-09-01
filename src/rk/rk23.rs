//! Bogacki–Shampine 3(2) pair (RK23) adaptive-step integrator.

use crate::{
    Float, ODE, SolOut, ControlFlag, Tolerance,
    rk::{RKSettings, RKResult},
    hinit::hinit,
    status::Status,
    error::Error,
    interpolate::Interpolate,
};

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
) -> Result<RKResult, Vec<Error>>
where
    F: ODE,
    S: SolOut,
    R: Into<Tolerance<'a>>,
    A: Into<Tolerance<'a>>,
{
    // --- Input Validation ---
    let mut errors: Vec<Error> = Vec::new();

    // Maximum Number of Steps
    let nmax = settings.nmax;
    if nmax <= 0 {
        errors.push(Error::NMaxMustBePositive(nmax));
    }

    // Safety Factor
    let safety_factor = settings.safety_factor;
    if safety_factor >= 1.0 || safety_factor <= 1e-4 {
        errors.push(Error::SafetyFactorOutOfRange(safety_factor));
    }

    // Step size scaling factors
    let scale_min = settings.scale_min;
    let scale_max = settings.scale_max;
    if scale_min <= 0.0 || scale_max <= scale_min {
        errors.push(Error::InvalidScaleFactors(scale_min, scale_max));
    }

    // Error exponent
    let error_exponent = settings.error_exponent;

    // Maximum step size
    let hmax = settings.h_max.map(|h| h.abs()).unwrap_or((xend - x).abs());

    // Initial step size: when not provided, use 0.0 so the core solver can set it
    let h = settings.h0.unwrap_or(0.0);

    if !errors.is_empty() {
        return Err(errors);
    }

    let rtol = rtol.into();
    let atol = atol.into();

    let result = rk23_core(f, x, y.to_vec(), xend, rtol, atol, solout, h, safety_factor, error_exponent, scale_min, scale_max, nmax, hmax);

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
    mut h: Float,
    safety_factor: Float,
    error_exponent: Float,
    scale_min: Float,
    scale_max: Float,
    nmax: usize,
    hmax: Float,
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
    let mut ye = vec![0.0; n];
    let mut q1 = vec![0.0; n];
    let mut q2 = vec![0.0; n];
    let mut q3 = vec![0.0; n];
    let mut nfev = 0;
    let mut nstep = 0;
    let mut naccpt = 0;
    let mut nrejct = 0;
    let mut status = Status::Success;
    let mut xold;
    let direction = (xend - x).signum();

    // Initial derivative
    f.ode(x, &y, &mut k1);
    nfev += 1;

    // Calculate initial step size if not provided
    if h == 0.0 {
        h = hinit(f, x, &y, direction, &k1, &mut k2, &mut yt, 3, hmax, &atol, &rtol);
        nfev += 1;
    }

    loop {
        if nstep >= nmax {
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
            ye[i] = h * (E1 * k1[i] + E2 * k2[i] + E3 * k3[i] + E4 * k4[i]);
        }

        // Error estimation
        let mut err = 0.0;
        for i in 0..n {
            let tol = atol[i] + rtol[i] * yt[i].abs().max(y[i].abs());
            err += (ye[i] / tol).powi(2);
        }
        err = (err / n as Float).sqrt();

        if err <= 1.0 {
            // Step accepted
            xold = x;
            ye.copy_from_slice(&y);
            x += h;
            y.copy_from_slice(&yt);
            nstep += 1;
            naccpt += 1;

            // Prepare dense output
            for i in 0..n {
                q1[i] = k1[i];
                q2[i] = D21 * k1[i] + D22 * k2[i] + D23 * k3[i] + D24 * k4[i];
                q3[i] = D31 * k1[i] + D32 * k2[i] + D33 * k3[i] + D34 * k4[i];
            }
            let dense_output = Rk23DenseOutput::new(h, xold, &ye, &q1, &q2, &q3);

            match solout.solout(xold, x, &y, &dense_output) {
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
            h *= (safety_factor * err.powf(error_exponent)).min(scale_max).max(scale_min);
            if h > hmax {
                h = hmax;
            }
        } else {
            // Step rejected
            nrejct += 1;
            h *= (safety_factor * err.powf(error_exponent)).min(1.0).max(scale_min);
        }
    }

    RKResult {
        x,
        y,
        h,
        nfev,
        nstep,
        naccpt,
        nrejct,
        status,
    }
}

/// Dense output interpolant for RK23
struct Rk23DenseOutput<'a> {
    t_old: Float,
    y_old: &'a [Float],
    q1: &'a [Float],
    q2: &'a [Float],
    q3: &'a [Float],
    h: Float,
}

impl<'a> Rk23DenseOutput<'a> {
    pub fn new(
        h: Float,
        t_old: Float,
        y_old: &'a [Float],
        q1: &'a [Float],
        q2: &'a [Float],
        q3: &'a [Float],
    ) -> Self {
        Self { t_old, y_old, q1, q2, q3, h }
    }
}

impl<'a> Interpolate for Rk23DenseOutput<'a> {
    fn interpolate(&self, ti: Float, yi: &mut [Float]) {
        let x = (ti - self.t_old) / self.h;
        let x2 = x * x;
        let x3 = x2 * x;
        for i in 0..yi.len() {
            yi[i] = self.y_old[i] + self.h * (self.q1[i] * x + self.q2[i] * x2 + self.q3[i] * x3);
        }
    }
}

// RK23 Butcher tableau coefficients
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

const D21: Float = -4.0 / 3.0;
const D22: Float = 1.0;
const D23: Float = 4.0 / 3.0;
const D24: Float = -1.0;
const D31: Float = 5.0 / 9.0;
const D32: Float = -2.0 / 3.0;
const D33: Float = -8.0 / 9.0;
const D34: Float = 1.0;
