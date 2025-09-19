//! DOPRI5 - Dormand–Prince 5(4) explicit Runge–Kutta integrator
//!
//! # Authors and attribution
//!
//! Translator / maintainer
//! - Ryan D. Gast <ryan.d.gast@gmail.com> (2025)
//!
//! Original authors
//! - E. Hairer and G. Wanner
//!   Université de Genève - Dept. de Mathématiques
//!   Emails: Ernst.Hairer@unige.ch, Gerhard.Wanner@unige.ch
//!
//! Reference
//! - E. Hairer, S. P. Nørsett, and G. Wanner, "Solving Ordinary Differential
//!   Equations I. Nonstiff Problems", 2nd ed., Springer (1993).
//!
//! Original Fortran implementation and supporting material
//! - https://www.unige.ch/~hairer/software.html
//!

use crate::{
    Float,
    error::Error,
    interpolate::Interpolate,
    methods::common::{Evals, IntegrationResult, Steps, Tolerance, hinit},
    ode::ODE,
    solout::{ControlFlag, SolOut},
    status::Status,
};

/// Dormand–Prince DOPRI5 — explicit embedded Runge–Kutta 5(4) solver with
/// adaptive step-size control and optional dense output.
///
/// This function integrates the autonomous system `y' = f(x, y)` from `x` to
/// `xend`, advancing the provided state buffer `y` in-place. It performs
/// classical error control (embedded estimates) and, optionally, computes
/// dense-output coefficients for continuous interpolation inside each step.
///
/// # Arguments
///
/// ## Defining the Problem
/// - `f`: Right‑hand side implementing `ODE`.
/// - `x`: Initial independent variable value.
/// - `xend`: Final independent variable value.
/// - `y`: Mutable slice containing the initial state; on success contains the
///   state at the final time.
/// - `rtol`, `atol`: Relative and absolute tolerances (see [`Tolerance`]).
///
/// ## Output Control
/// - `solout`: Optional mutable reference to a `SolOut` callback used for
///   intermediate output and event handling. If `dense_output` is `true` the
///   callback may receive a dense interpolant.
/// - `dense_output`: If `true`, dense‑output coefficients are computed every
///   accepted step to enable fast interpolation via the provided interpolant.
///
/// ## Optional Settings
///
/// Below are optional parameters to customize the integrator's settings.
/// If `None` is provided the default value is used. The default values
/// should be suitable for most problems.
///
/// - `uround` (default `2.3e-16`)
/// - `safety_factor` (default `0.9`)
/// - `scale_min` (default sets `facc1 = 5`)
/// - `scale_max` (default sets `facc2 = 1/10`)
/// - `beta` (default `0.04` — stabilization parameter)
/// - `h_max` (default `|xend - x|`)
/// - `h0` (initial step; heuristic if `None`)
/// - `max_steps` (default `100_000`)
/// - `stiff_test` (steps between stiffness tests, default `1000`)
///
/// # Returns
/// A `Result` with `IntegrationResult` on success or a vector of `Error`
/// values describing input validation issues.
pub fn dopri5<F, S>(
    f: &F,
    mut x: Float,
    xend: Float,
    y: &mut [Float],
    rtol: Tolerance,
    atol: Tolerance,
    mut solout: Option<&mut S>,
    dense_output: bool,
    uround: Option<Float>,
    safety_factor: Option<Float>,
    scale_min: Option<Float>,
    scale_max: Option<Float>,
    beta: Option<Float>,
    h_max: Option<Float>,
    h0: Option<Float>,
    max_steps: Option<usize>,
    stiff_test: Option<usize>,
) -> Result<IntegrationResult, Vec<Error>>
where
    F: ODE,
    S: SolOut,
{
    // --- Input Validation ---
    let mut errors: Vec<Error> = Vec::new();

    // Rounding Unit
    let uround = match uround {
        Some(u) => {
            if u <= 1e-35 || u >= 1.0 {
                errors.push(Error::URoundOutOfRange(u));
            }
            u
        }
        None => 2.3e-16,
    };

    // Safety Factor
    let safety_factor = match safety_factor {
        Some(f) => {
            if f >= 1.0 || f <= 1e-4 {
                errors.push(Error::SafetyFactorOutOfRange(f));
            }
            f
        }
        None => 0.9,
    };

    // Parameters for step size selection
    let facc1 = match scale_min {
        Some(f) => 1.0 / f,
        None => 5.0,
    };
    let facc2 = match scale_max {
        Some(f) => 1.0 / f,
        None => 1.0 / 10.0,
    };

    // Beta for step control stabilization
    let beta = match beta {
        Some(b) => {
            if b > 0.2 {
                errors.push(Error::BetaTooLarge(b));
            }
            b
        }
        None => 0.04,
    };

    // Maximum step size
    let h_max = match h_max {
        Some(h) => h.abs(),
        None => (xend - x).abs(),
    };

    // Maximum Number of Steps
    let nmax = match max_steps {
        Some(n) => {
            if n <= 0 {
                errors.push(Error::NMaxMustBePositive(n));
            }
            n
        }
        None => 100_000,
    };

    // Number of steps before performing a stiffness test
    let nstiff = match stiff_test {
        Some(n) => {
            if n <= 0 {
                errors.push(Error::NStiffMustBePositive(n));
            }
            n
        }
        None => 1000,
    };

    if !errors.is_empty() {
        return Err(errors);
    }

    // --- Declarations ---
    let n = y.len();
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut k5 = vec![0.0; n];
    let mut k6 = vec![0.0; n];
    let mut y1 = vec![0.0; n];
    let mut cont = vec![0.0; n * 5];
    let mut facold: Float = 1e-4;
    let mut last = false;
    let mut reject = false;
    let mut nonstiff = 0;
    let mut hlamb = 0.0;
    let mut iasti = 0;
    let mut fac11;
    let mut fac;
    let mut hnew;
    let mut xph;
    let mut evals = Evals::new();
    let mut steps = Steps::new();
    let mut xold = x;
    let mut xout = None;
    let mut event;
    let status;
    let expo1 = 0.2 - beta * 0.75;
    let posneg = (xend - x).signum();

    // --- Initializations ---
    f.ode(x, &y, &mut k1);
    evals.ode += 1;
    let mut h = match h0 {
        Some(h0) => h0.abs() * posneg,
        None => {
            evals.ode += 1;
            hinit(
                f, x, &y, posneg, &k1, &mut k2, &mut y1, 5, h_max, &atol, &rtol,
            )
        }
    };

    // Interpolator for optional dense output to SolOut
    let interpolator = &DenseOutput::new(
        cont.as_ptr(),
        cont.len(),
        &xold as *const Float,
        &h as *const Float,
    );

    // Initial SolOut call
    if let Some(solout) = solout.as_mut() {
        match solout.solout::<DenseOutput>(xold, x, &y, None) {
            ControlFlag::Interrupt => {
                return Ok(IntegrationResult {
                    x,
                    h,
                    status: Status::UserInterrupt,
                    evals,
                    steps,
                });
            }
            ControlFlag::ModifiedSolution(xm, ym) => {
                x = xm;
                y.copy_from_slice(&ym);
                f.ode(x, &y, &mut k1);
                evals.ode += 1;
            }
            ControlFlag::XOut(xo) => {
                xout = Some(xo);
            }
            ControlFlag::Continue => {}
        }
    }

    // --- Main integration loop ---
    loop {
        // Check for maximum number of steps
        if steps.total > nmax {
            status = Status::NeedLargerNMax;
            break;
        }

        // Check for underflow due to machine rounding
        if 0.1 * h.abs() <= x.abs() * uround {
            status = Status::StepSizeTooSmall;
            break;
        }

        // Adjust last step to land on xend
        if (x + 1.01 * h - xend) * posneg > 0.0 {
            h = xend - x;
            last = true;
        }

        steps.total += 1;

        // Stage 2
        for i in 0..n {
            y1[i] = y[i] + h * A21 * k1[i];
        }
        f.ode(x + C2 * h, &y1, &mut k2);

        // Stage 3
        for i in 0..n {
            y1[i] = y[i] + h * (A31 * k1[i] + A32 * k2[i]);
        }
        f.ode(x + C3 * h, &y1, &mut k3);

        // Stage 4
        for i in 0..n {
            y1[i] = y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]);
        }
        f.ode(x + C4 * h, &y1, &mut k4);

        // Stage 5
        for i in 0..n {
            y1[i] = y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]);
        }
        f.ode(x + C5 * h, &y1, &mut k5);

        // Stage 6 (ysti)
        for i in 0..n {
            y1[i] =
                y[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]);
        }
        xph = x + h;
        f.ode(xph, &y1, &mut k6);

        // Final stage
        for i in 0..n {
            y1[i] =
                y[i] + h * (A71 * k1[i] + A73 * k3[i] + A74 * k4[i] + A75 * k5[i] + A76 * k6[i]);
        }
        f.ode(xph, &y1, &mut k2);
        evals.ode += 6;

        // Prepare last segment of dense output before recalculating k4
        event = xout.map_or(false, |xo| xo <= xph);
        if dense_output || event {
            for i in 0..n {
                cont[4 * n + i] = h
                    * (D1 * k1[i] + D3 * k3[i] + D4 * k4[i] + D5 * k5[i] + D6 * k6[i] + D7 * k2[i]);
            }
        }

        // K4 scaled for error estimate
        for i in 0..n {
            k4[i] =
                (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k2[i]) * h;
        }

        // Error estimation
        let mut err = 0.0_f64;
        for i in 0..n {
            let sk = atol[i] + rtol[i] * y[i].abs().max(y1[i].abs());
            err += (k4[i] / sk) * (k4[i] / sk);
        }
        err = (err / n as f64).sqrt();

        // Computation of hnew
        fac11 = err.powf(expo1);
        // Lund-Stabilization
        fac = fac11 / facold.powf(beta);
        // We require fac1 <= hnew/h <= fac2
        fac = facc2.max(facc1.min(fac / safety_factor));
        hnew = h / fac;

        if err <= 1.0 {
            // Step accepted
            facold = err.max(1.0e-4);
            steps.accepted += 1;

            // Stiffness detection
            if (steps.accepted % nstiff == 0) || (iasti > 0) {
                let mut stnum = 0.0_f64;
                let mut stden = 0.0_f64;
                for i in 0..n {
                    let d1 = k2[i] - k6[i];
                    let ysti = y[i]
                        + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]);
                    let d2 = y1[i] - ysti;
                    stnum += d1 * d1;
                    stden += d2 * d2;
                }
                if stden > 0.0 {
                    hlamb = h.abs() * (stnum / stden).sqrt();
                }
                if hlamb > 3.25 {
                    nonstiff = 0;
                    iasti += 1;
                    if iasti == 15 {
                        status = Status::ProbablyStiff;
                        break;
                    }
                } else {
                    nonstiff += 1;
                    if nonstiff == 6 {
                        iasti = 0;
                    }
                }
            }

            // Prepare dense output
            if dense_output || event {
                for i in 0..n {
                    let ydiff = y1[i] - y[i];
                    let bspl = h * k1[i] - ydiff;
                    cont[i] = y[i];
                    cont[n + i] = ydiff;
                    cont[2 * n + i] = bspl;
                    cont[3 * n + i] = -h * k2[i] + ydiff - bspl;
                }
            }

            // Update state variables
            k1.copy_from_slice(&k2);
            y.copy_from_slice(&y1);
            xold = x;
            x = xph;

            if let Some(solout) = solout.as_mut() {
                let interpolation = if dense_output || event {
                    Some(interpolator)
                } else {
                    None
                };
                match solout.solout(xold, x, &y, interpolation) {
                    ControlFlag::Interrupt => {
                        status = Status::UserInterrupt;
                        break;
                    }
                    ControlFlag::ModifiedSolution(xm, ym) => {
                        x = xm;
                        y.copy_from_slice(&ym);
                        f.ode(x, &y, &mut k2);
                        evals.ode += 1;
                    }
                    ControlFlag::XOut(xo) => {
                        xout = Some(xo);
                    }
                    ControlFlag::Continue => {}
                }
            }

            // Normal exit
            if last {
                h = hnew;
                status = Status::Success;
                break;
            }

            // Check for step size limits
            if hnew.abs() > h_max.abs() {
                hnew = posneg * h_max.abs();
            }

            // Prevent oscillations due to previous rejected step
            if reject {
                hnew = posneg * hnew.abs().min(h.abs());
                reject = false;
            }
        } else {
            // Step rejected
            hnew = h / facc1.min(fac11 / safety_factor);
            reject = true;
            if steps.accepted > 1 {
                steps.rejected += 1;
            }
            last = false;
        }
        h = hnew;
    }

    Ok(IntegrationResult::new(x, h, status, evals, steps))
}

/// Continuous output function for DOPRI5
pub fn contdp5(xi: Float, yi: &mut [Float], cont: &[Float], xold: Float, h: Float) {
    let n = cont.len() / 5;
    let theta = (xi - xold) / h;
    let theta1 = 1.0 - theta;
    for i in 0..n {
        yi[i] = cont[i]
            + theta
                * (cont[n + i]
                    + theta1
                        * (cont[2 * n + i] + theta * (cont[3 * n + i] + theta1 * cont[4 * n + i])));
    }
}

/// Dense output interpolator for DOPRI5
struct DenseOutput {
    cont_ptr: *const Float,
    cont_len: usize,
    xold_ptr: *const Float,
    h_ptr: *const Float,
}

impl DenseOutput {
    fn new(
        cont_ptr: *const Float,
        cont_len: usize,
        xold_ptr: *const Float,
        h_ptr: *const Float,
    ) -> Self {
        Self {
            cont_ptr,
            cont_len,
            xold_ptr,
            h_ptr,
        }
    }
}

impl Interpolate for DenseOutput {
    fn interpolate(&self, xi: Float, yi: &mut [Float]) {
        unsafe {
            let cont_slice = std::slice::from_raw_parts(self.cont_ptr, self.cont_len);
            let xold = *self.xold_ptr;
            let h = *self.h_ptr;
            contdp5(xi, yi, cont_slice, xold, h);
        }
    }

    fn get_cont(&self) -> (Vec<Float>, Float, Float) {
        unsafe {
            let cont_slice = std::slice::from_raw_parts(self.cont_ptr, self.cont_len);
            let xold = *self.xold_ptr;
            let h = *self.h_ptr;
            (cont_slice.to_vec(), xold, h)
        }
    }
}

// DOPRI5 Butcher tableau coefficients
const C2: Float = 0.2;
const C3: Float = 0.3;
const C4: Float = 0.8;
const C5: Float = 8.0 / 9.0;

const A21: Float = 0.2;
const A31: Float = 3.0 / 40.0;
const A32: Float = 9.0 / 40.0;
const A41: Float = 44.0 / 45.0;
const A42: Float = -56.0 / 15.0;
const A43: Float = 32.0 / 9.0;
const A51: Float = 19372.0 / 6561.0;
const A52: Float = -25360.0 / 2187.0;
const A53: Float = 64448.0 / 6561.0;
const A54: Float = -212.0 / 729.0;
const A61: Float = 9017.0 / 3168.0;
const A62: Float = -355.0 / 33.0;
const A63: Float = 46732.0 / 5247.0;
const A64: Float = 49.0 / 176.0;
const A65: Float = -5103.0 / 18656.0;
const A71: Float = 35.0 / 384.0;
const A73: Float = 500.0 / 1113.0;
const A74: Float = 125.0 / 192.0;
const A75: Float = -2187.0 / 6784.0;
const A76: Float = 11.0 / 84.0;

const E1: Float = 71.0 / 57600.0;
const E3: Float = -71.0 / 16695.0;
const E4: Float = 71.0 / 1920.0;
const E5: Float = -17253.0 / 339200.0;
const E6: Float = 22.0 / 525.0;
const E7: Float = -1.0 / 40.0;

const D1: Float = -12715105075.0 / 11282082432.0;
const D3: Float = 87487479700.0 / 32700410799.0;
const D4: Float = -10690763975.0 / 1880347072.0;
const D5: Float = 701980252875.0 / 199316789632.0;
const D6: Float = -1453857185.0 / 822651844.0;
const D7: Float = 69997945.0 / 29380423.0;
