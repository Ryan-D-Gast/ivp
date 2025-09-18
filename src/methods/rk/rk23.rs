//! Bogacki–Shampine 3(2) pair (RK23) adaptive-step integrator.

use crate::{
    Float,
    error::Error,
    interpolate::Interpolate,
    methods::{
        hinit::hinit,
        result::{Evals, IntegrationResult, Steps},
        settings::{Settings, Tolerance},
    },
    ode::ODE,
    solout::{ControlFlag, SolOut},
    status::Status,
};

/// Bogacki–Shampine 3(2) pair (RK23) adaptive-step integrator.
/// This implementation uses an embedded method to estimate errors
/// and adjust the step size accordingly with dense output.
pub fn rk23<F, S>(
    f: &F,
    mut x: Float,
    xend: Float,
    y: &mut [Float],
    rtol: Tolerance,
    atol: Tolerance,
    solout: &mut S,
    settings: Settings,
) -> Result<IntegrationResult, Vec<Error>>
where
    F: ODE,
    S: SolOut,
{
    // --- Input Validation ---
    let mut errors: Vec<Error> = Vec::new();

    // Maximum Number of Steps
    let nmax = match settings.nmax {
        Some(n) => {
            if n <= 0 {
                errors.push(Error::NMaxMustBePositive(n));
            }
            n
        }
        None => 100_000,
    };

    // Safety Factor
    let safety_factor = match settings.safety_factor {
        Some(f) => {
            if f >= 1.0 || f <= 1e-4 {
                errors.push(Error::SafetyFactorOutOfRange(f));
            }
            f
        }
        None => 0.9,
    };

    // Step size scaling factors
    let scale_min = settings.scale_min.unwrap_or(0.2);
    let scale_max = settings.scale_max.unwrap_or(5.0);
    if scale_min <= 0.0 || scale_max <= scale_min {
        errors.push(Error::InvalidScaleFactors(scale_min, scale_max));
    }

    // Error exponent
    let error_exponent = -1.0 / 3.0;

    // Maximum step size
    let hmax = settings.hmax.map(|h| h.abs()).unwrap_or((xend - x).abs());

    // Set SolOut calling behavior
    let solout_flag = settings.solout_flag;

    if !errors.is_empty() {
        return Err(errors);
    }

    // --- Declarations ---
    let n = y.len();
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut yt = vec![0.0; n];
    let mut ye = vec![0.0; n];
    let mut cont = vec![0.0; 4 * n];
    let mut evals = Evals::new();
    let mut steps = Steps::new();
    let mut status = Status::Success;
    let mut xold = x;
    let direction = (xend - x).signum();

    // --- Initializations ---
    f.ode(x, &y, &mut k1);
    evals.ode += 1;
    let mut h = match settings.h0 {
        Some(h0) => h0,
        None => {
            evals.ode += 1;
            hinit(
                f, x, &y, direction, &k1, &mut k2, &mut k3, 3, hmax, &atol, &rtol,
            )
        }
    };
    if solout_flag.call() {
        cont[0..n].copy_from_slice(&y);
        let interp = DenseOutput::new(&cont, xold, h);
        solout.solout(xold, x, &y, &interp);
    }

    // --- Main integration loop ---
    loop {
        // Check for maximum number of steps
        if steps.total >= nmax {
            status = Status::NeedLargerNMax;
            break;
        }

        // Check for last step adjustment
        if (x + h - xend) * direction > 0.0 {
            h = xend - x;
        }

        // Stage 2
        for i in 0..n {
            yt[i] = y[i] + h * A21 * k1[i];
        }
        f.ode(x + C2 * h, &yt, &mut k2);

        // Stage 3
        for i in 0..n {
            yt[i] = y[i] + h * A32 * k2[i];
        }
        f.ode(x + C3 * h, &yt, &mut k3);

        // Compute solution and error estimate
        for i in 0..n {
            yt[i] = y[i] + h * (B1 * k1[i] + B2 * k2[i] + B3 * k3[i]);
        }

        // Stage 4/1: derivative at new point, also used as k1 if accepted.
        f.ode(x + h, &yt, &mut k4);

        evals.ode += 3;

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
            steps.total += 1;
            steps.accepted += 1;

            // Update state
            ye.copy_from_slice(&y);
            y.copy_from_slice(&yt);
            xold = x;
            x += h;

            // Prepare dense output
            if solout_flag.dense() {
                cont[0..n].copy_from_slice(&ye);
                for i in 0..n {
                    cont[n + i] = k1[i];
                    cont[2 * n + i] = D21 * k1[i] + D22 * k2[i] + D23 * k3[i] + D24 * k4[i];
                    cont[3 * n + i] = D31 * k1[i] + D32 * k2[i] + D33 * k3[i] + D34 * k4[i];
                }
            }

            // Optional callback function
            if solout_flag.call() {
                match solout.solout(xold, x, &y, &DenseOutput::new(&cont, xold, h)) {
                    ControlFlag::Interrupt => {
                        status = Status::UserInterrupt;
                        break;
                    }
                    ControlFlag::ModifiedSolution(xm, ym) => {
                        // Update with modified solution
                        x = xm;
                        for i in 0..n {
                            y[i] = ym[i];
                        }

                        // Recompute k1 at new (x, y).
                        f.ode(x, &y, &mut k1);
                        evals.ode += 1;
                    }
                    ControlFlag::Continue => {
                        // Reuse k4 as k1 for the next step to save an evaluation.
                        k1.copy_from_slice(&k4);
                    }
                }
            }

            // Adjust step size
            h *= (safety_factor * err.powf(error_exponent))
                .min(scale_max)
                .max(scale_min);
            if h > hmax {
                h = hmax;
            }

            // Normal exit
            if x == xend {
                break;
            }
        } else {
            // Step rejected
            steps.rejected += 1;
            h *= (safety_factor * err.powf(error_exponent))
                .min(1.0)
                .max(scale_min);
        }
    }

    Ok(IntegrationResult::new(x, h, status, evals, steps))
}

/// Dense output evaluation for RK23
pub fn contrk23(xi: Float, yi: &mut [Float], cont: &[Float], xold: Float, h: Float) {
    let n = yi.len();
    let x = (xi - xold) / h;
    let x2 = x * x;
    let x3 = x2 * x;
    for i in 0..n {
        yi[i] = cont[i] + h * (cont[n + i] * x + cont[2 * n + i] * x2 + cont[3 * n + i] * x3);
    }
}

struct DenseOutput<'a> {
    cont: &'a [Float],
    xold: Float,
    h: Float,
}

impl<'a> DenseOutput<'a> {
    pub fn new(cont: &'a [Float], xold: Float, h: Float) -> Self {
        Self { xold, cont, h }
    }
}

impl<'a> Interpolate for DenseOutput<'a> {
    fn interpolate(&self, ti: Float, yi: &mut [Float]) {
        contrk23(ti, yi, self.cont, self.xold, self.h);
    }

    fn get_cont(&self) -> (Vec<Float>, Float, Float) {
        (self.cont.to_vec(), self.xold, self.h)
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
