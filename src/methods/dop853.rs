//! DOP853 - Dormand–Prince 8(5,3) explicit Runge–Kutta integrator
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
    methods::{Evals, IntegrationResult, Steps, Tolerance, hinit},
    ode::ODE,
    solout::{ControlFlag, SolOut},
    status::Status,
};

/// DOP853 - Dormand–Prince 8(5,3) explicit Runge–Kutta integrator.
#[derive(Clone, Copy, Debug)]
pub struct DOP853;

impl DOP853 {
    /// Dormand–Prince DOP853 — explicit Runge–Kutta 8(5,3) solver with adaptive
    /// step-size control and optional dense output.
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
    ///   callback may receive a dense interpolant; otherwise interpolation is
    ///   computed only when needed to satisfy a previously requested `xout`.
    /// - `dense_output`: If `true`, dense‑output coefficients are computed every
    ///   accepted step to enable fast interpolation via the provided interpolant.
    ///
    /// ## Optional Settings
    ///
    /// Below are optional parameters to customize the integrator's settings.
    /// If `None` is provided the default value is used. The default values
    /// should be suitable for most problems.
    ///
    /// - `uround`: Machine rounding unit (default `2.3e-16`).
    /// - `safety_factor`: Safety factor for stepsize control (default `0.9`).
    /// - `scale_min`, `scale_max`: Lower/upper bounds for the stepsize scale.
    /// - `beta`: Stabilization parameter for step control (default `0.0`).
    /// - `h_max`: Maximum allowed step size (default `|xend - x|`).
    /// - `h0`: Optional initial step-size guess; when `None` a heuristic is used.
    /// - `max_steps`: Maximum number of steps (default `100_000`).
    /// - `stiff_test`: Number of accepted steps between stiffness tests
    ///   (default `1000`).
    ///
    /// # Returns
    /// `Result<IntegrationResult, Vec<Error>>` with [`IntegrationResult`] containing
    /// the final `x`, the predicted next step size `h`, the [`Status`] of the solver,
    /// the [`Evals`] and [`Steps`] statistics of the solver.
    ///
    /// On error, a vector of [`Error`] values describing input validation issues.
    pub fn solve<F, S>(
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
            None => 3.0,
        };
        let facc2 = match scale_max {
            Some(f) => 1.0 / f,
            None => 1.0 / 6.0,
        };

        // Beta for step control stabilization
        let beta = match beta {
            Some(b) => {
                if b > 0.2 {
                    errors.push(Error::BetaTooLarge(b));
                }
                b
            }
            None => 0.00,
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
        let mut y1 = vec![0.0; n];
        let mut k1 = vec![0.0; n];
        let mut k2 = vec![0.0; n];
        let mut k3 = vec![0.0; n];
        let mut k4 = vec![0.0; n];
        let mut k5 = vec![0.0; n];
        let mut k6 = vec![0.0; n];
        let mut k7 = vec![0.0; n];
        let mut k8 = vec![0.0; n];
        let mut k9 = vec![0.0; n];
        let mut k10 = vec![0.0; n];
        let mut cont = vec![0.0; n * 8];
        let mut nonstiff = 0;
        let mut facold: Float = 1e-4;
        let mut hlamb = 0.0;
        let mut iasti = 0;
        let mut err;
        let mut err2;
        let mut deno;
        let mut fac;
        let mut hnew;
        let mut fac11;
        let mut sk;
        let mut erri;
        let mut xph;
        let mut last = false;
        let mut reject = false;
        let mut evals = Evals::new();
        let mut steps = Steps::new();
        let mut xold = x;
        let mut xout = None;
        let mut event;
        let expo1 = 1.0 / 8.0 - beta * 0.2;
        let status;
        let posneg = (xend - x).signum();

        // --- Initializations ---
        f.ode(x, &y, &mut k1);
        evals.ode += 1;
        let mut h = match h0 {
            Some(h0) => h0.abs() * posneg,
            None => {
                evals.ode += 1;
                hinit(
                    f, x, &y, posneg, &k1, &mut k2, &mut y1, 8, h_max, &atol, &rtol,
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

        // Initial call to SolOut
        if let Some(solout) = solout.as_mut() {
            match solout.solout::<DenseOutput>(xold, x, &y, None) {
                ControlFlag::Interrupt => {
                    status = Status::UserInterrupt;
                    return Ok(IntegrationResult {
                        x,
                        h,
                        status,
                        evals,
                        steps,
                    });
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
                ControlFlag::XOut(xo) => {
                    // Update x to xout
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

            // --- The twelve stages ---
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
                y1[i] = y[i] + h * (A41 * k1[i] + A43 * k3[i]);
            }
            f.ode(x + C4 * h, &y1, &mut k4);

            // Stage 5
            for i in 0..n {
                y1[i] = y[i] + h * (A51 * k1[i] + A53 * k3[i] + A54 * k4[i]);
            }
            f.ode(x + C5 * h, &y1, &mut k5);

            // Stage 6
            for i in 0..n {
                y1[i] = y[i] + h * (A61 * k1[i] + A64 * k4[i] + A65 * k5[i]);
            }
            f.ode(x + C6 * h, &y1, &mut k6);

            // Stage 7
            for i in 0..n {
                y1[i] = y[i] + h * (A71 * k1[i] + A74 * k4[i] + A75 * k5[i] + A76 * k6[i]);
            }
            f.ode(x + C7 * h, &y1, &mut k7);

            // Stage 8
            for i in 0..n {
                y1[i] =
                    y[i] + h * (A81 * k1[i] + A84 * k4[i] + A85 * k5[i] + A86 * k6[i] + A87 * k7[i]);
            }
            f.ode(x + C8 * h, &y1, &mut k8);

            // Stage 9
            for i in 0..n {
                y1[i] = y[i]
                    + h * (A91 * k1[i]
                        + A94 * k4[i]
                        + A95 * k5[i]
                        + A96 * k6[i]
                        + A97 * k7[i]
                        + A98 * k8[i]);
            }
            f.ode(x + C9 * h, &y1, &mut k9);

            // Stage 10
            for i in 0..n {
                y1[i] = y[i]
                    + h * (A101 * k1[i]
                        + A104 * k4[i]
                        + A105 * k5[i]
                        + A106 * k6[i]
                        + A107 * k7[i]
                        + A108 * k8[i]
                        + A109 * k9[i]);
            }
            f.ode(x + C10 * h, &y1, &mut k10);

            // Stage 11
            for i in 0..n {
                y1[i] = y[i]
                    + h * (A111 * k1[i]
                        + A114 * k4[i]
                        + A115 * k5[i]
                        + A116 * k6[i]
                        + A117 * k7[i]
                        + A118 * k8[i]
                        + A119 * k9[i]
                        + A1110 * k10[i]);
            }
            f.ode(x + C11 * h, &y1, &mut k2);

            // Stage 12
            xph = x + h;
            for i in 0..n {
                y1[i] = y[i]
                    + h * (A121 * k1[i]
                        + A124 * k4[i]
                        + A125 * k5[i]
                        + A126 * k6[i]
                        + A127 * k7[i]
                        + A128 * k8[i]
                        + A129 * k9[i]
                        + A1210 * k10[i]
                        + A1211 * k2[i]);
            }
            f.ode(xph, &y1, &mut k3);
            evals.ode += 11;

            for i in 0..n {
                k4[i] = B1 * k1[i]
                    + B6 * k6[i]
                    + B7 * k7[i]
                    + B8 * k8[i]
                    + B9 * k9[i]
                    + B10 * k10[i]
                    + B11 * k2[i]
                    + B12 * k3[i];
                k5[i] = y[i] + h * k4[i];
            }

            // Error estimation
            err = 0.0;
            err2 = 0.0;
            for i in 0..n {
                sk = atol[i] + rtol[i] * y[i].abs().max(k5[i].abs());

                // ERR2 uses K4 - BHH1*K1 - BHH2*K9 - BHH3*K3
                erri = k4[i] - BH1 * k1[i] - BH2 * k9[i] - BH3 * k3[i];
                err2 += (erri / sk).powi(2);

                // ERRI = er1*K1 + er6*K6 + er7*K7 + er8*K8 + er9*K9 + er10*K10 + er11*K2 + er12*K3
                erri = ER1 * k1[i]
                    + ER6 * k6[i]
                    + ER7 * k7[i]
                    + ER8 * k8[i]
                    + ER9 * k9[i]
                    + ER10 * k10[i]
                    + ER11 * k2[i]
                    + ER12 * k3[i];
                err += (erri / sk).powi(2);
            }
            deno = err + 0.01 * err2;
            if deno <= 0.0 {
                deno = 1.0;
            }
            err = h.abs() * err * (1.0 / (n as f64 * deno)).sqrt();

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
                f.ode(xph, &k5, &mut k4);
                evals.ode += 1;

                // Stiffness detection
                if (steps.accepted % nstiff == 0) || (iasti > 0) {
                    let mut stnum: Float = 0.0;
                    let mut stden: Float = 0.0;
                    for i in 0..n {
                        let d1 = k4[i] - k3[i];
                        let d2 = k5[i] - y1[i];
                        stnum += d1 * d1;
                        stden += d2 * d2;
                    }
                    if stden > 0.0 {
                        hlamb = h.abs() * (stnum / stden).sqrt();
                    }
                    if hlamb > 6.1 {
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
                event = xout.map_or(false, |xo| xo <= xph);
                if dense_output || event {
                    for i in 0..n {
                        cont[i] = y[i];
                        let ydiff = k5[i] - y[i];
                        cont[n + i] = ydiff;
                        let bspl = h * k1[i] - ydiff;
                        cont[2 * n + i] = bspl;
                        cont[3 * n + i] = ydiff - h * k4[i] - bspl;
                        cont[4 * n + i] = D41 * k1[i]
                            + D46 * k6[i]
                            + D47 * k7[i]
                            + D48 * k8[i]
                            + D49 * k9[i]
                            + D410 * k10[i]
                            + D411 * k2[i]
                            + D412 * k3[i];

                        cont[5 * n + i] = D51 * k1[i]
                            + D56 * k6[i]
                            + D57 * k7[i]
                            + D58 * k8[i]
                            + D59 * k9[i]
                            + D510 * k10[i]
                            + D511 * k2[i]
                            + D512 * k3[i];

                        cont[6 * n + i] = D61 * k1[i]
                            + D66 * k6[i]
                            + D67 * k7[i]
                            + D68 * k8[i]
                            + D69 * k9[i]
                            + D610 * k10[i]
                            + D611 * k2[i]
                            + D612 * k3[i];

                        cont[7 * n + i] = D71 * k1[i]
                            + D76 * k6[i]
                            + D77 * k7[i]
                            + D78 * k8[i]
                            + D79 * k9[i]
                            + D710 * k10[i]
                            + D711 * k2[i]
                            + D712 * k3[i];
                    }

                    // Next three function evaluations
                    for i in 0..n {
                        y1[i] = y[i]
                            + h * (A141 * k1[i]
                                + A147 * k7[i]
                                + A148 * k8[i]
                                + A149 * k9[i]
                                + A1410 * k10[i]
                                + A1411 * k2[i]
                                + A1412 * k3[i]
                                + A1413 * k4[i]);
                    }
                    f.ode(x + C14 * h, &y1, &mut k10);

                    for i in 0..n {
                        y1[i] = y[i]
                            + h * (A151 * k1[i]
                                + A156 * k6[i]
                                + A157 * k7[i]
                                + A158 * k8[i]
                                + A1511 * k2[i]
                                + A1512 * k3[i]
                                + A1513 * k4[i]
                                + A1514 * k10[i]);
                    }
                    f.ode(x + C15 * h, &y1, &mut k2);

                    for i in 0..n {
                        y1[i] = y[i]
                            + h * (A161 * k1[i]
                                + A166 * k6[i]
                                + A167 * k7[i]
                                + A168 * k8[i]
                                + A169 * k9[i]
                                + A1613 * k4[i]
                                + A1614 * k10[i]
                                + A1615 * k2[i]);
                    }
                    f.ode(x + C16 * h, &y1, &mut k3);
                    evals.ode += 3;

                    // Add contributions of last three stages
                    for i in 0..n {
                        cont[4 * n + i] = h
                            * (cont[4 * n + i]
                                + D413 * k4[i]
                                + D414 * k10[i]
                                + D415 * k2[i]
                                + D416 * k3[i]);

                        cont[5 * n + i] = h
                            * (cont[5 * n + i]
                                + D513 * k4[i]
                                + D514 * k10[i]
                                + D515 * k2[i]
                                + D516 * k3[i]);

                        cont[6 * n + i] = h
                            * (cont[6 * n + i]
                                + D613 * k4[i]
                                + D614 * k10[i]
                                + D615 * k2[i]
                                + D616 * k3[i]);

                        cont[7 * n + i] = h
                            * (cont[7 * n + i]
                                + D713 * k4[i]
                                + D714 * k10[i]
                                + D715 * k2[i]
                                + D716 * k3[i]);
                    }
                }

                // Update state variables
                k1.copy_from_slice(&k4);
                y.copy_from_slice(&k5);
                xold = x;
                x = xph;

                // Call to SolOut
                if let Some(solout) = solout.as_mut() {
                    // See if interpolation is provided
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
                            // Update with modified solution
                            x = xm;
                            y.copy_from_slice(&ym);

                            // Recompute k4 at new (x, y)
                            f.ode(x, &y, &mut k4);
                            evals.ode += 1;
                        }
                        ControlFlag::XOut(xo) => {
                            // Update xout
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

    /// Continuous output function for DOP853
    pub fn interpolate(xi: Float, yi: &mut [Float], cont: &[Float], xold: Float, h: Float) {
        let n = cont.len() / 8;
        let s = (xi - xold) / h;
        let s1 = 1.0 - s;
        for i in 0..n {
            let conpar =
                cont[4 * n + i] + s * (cont[5 * n + i] + s1 * (cont[6 * n + i] + s * cont[7 * n + i]));
            let contd8 = cont[i]
                + s * (cont[n + i] + s1 * (cont[2 * n + i] + s * (cont[3 * n + i] + s1 * conpar)));
            yi[i] = contd8;
        }
    }
}

/// Dense output interpolator for DOP853
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

impl<'a> Interpolate for DenseOutput {
    fn interpolate(&self, xi: Float, yi: &mut [Float]) {
        unsafe {
            let cont_slice = std::slice::from_raw_parts(self.cont_ptr, self.cont_len);
            let xold = *self.xold_ptr;
            let h = *self.h_ptr;
            DOP853::interpolate(xi, yi, cont_slice, xold, h);
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

// DOP853 Butcher tableau coefficients
const C2: Float = 0.526001519587677318785587544488e-01;
const C3: Float = 0.789002279381515978178381316732e-01;
const C4: Float = 0.118350341907227396726757197510e+00;
const C5: Float = 0.281649658092772603273242802490e+00;
const C6: Float = 0.333333333333333333333333333333e+00;
const C7: Float = 0.25e+00;
const C8: Float = 0.307692307692307692307692307692e+00;
const C9: Float = 0.651282051282051282051282051282e+00;
const C10: Float = 0.6e+00;
const C11: Float = 0.857142857142857142857142857142e+00;
const C14: Float = 0.1e+00;
const C15: Float = 0.2e+00;
const C16: Float = 7.777_777_777_777_778e-1;

const A21: Float = 5.26001519587677318785587544488e-2;

const A31: Float = 1.97250569845378994544595329183e-2;
const A32: Float = 5.91751709536136983633785987549e-2;

const A41: Float = 2.95875854768068491816892993775e-2;
const A43: Float = 8.87627564304205475450678981324e-2;

const A51: Float = 2.41365134159266685502369798665e-1;
const A53: Float = -8.84549479328286085344864962717e-1;
const A54: Float = 9.24834003261792003115737966543e-1;

const A61: Float = 3.7037037037037037037037037037e-2;
const A64: Float = 1.70828608729473871279604482173e-1;
const A65: Float = 1.25467687566822425016691814123e-1;

const A71: Float = 3.7109375e-2;
const A74: Float = 1.70252211019544039314978060272e-1;
const A75: Float = 6.02165389804559606850219397283e-2;
const A76: Float = -1.7578125e-2;

const A81: Float = 3.70920001185047927108779319836e-2;
const A84: Float = 1.70383925712239993810214054705e-1;
const A85: Float = 1.07262030446373284651809199168e-1;
const A86: Float = -1.53194377486244017527936158236e-2;
const A87: Float = 8.27378916381402288758473766002e-3;

const A91: Float = 6.24110958716075717114429577812e-1;
const A94: Float = -3.36089262944694129406857109825e0;
const A95: Float = -8.68219346841726006818189891453e-1;
const A96: Float = 2.75920996994467083049415600797e1;
const A97: Float = 2.01540675504778934086186788979e1;
const A98: Float = -4.34898841810699588477366255144e1;

const A101: Float = 4.77662536438264365890433908527e-1;
const A104: Float = -2.48811461997166764192642586468e0;
const A105: Float = -5.90290826836842996371446475743e-1;
const A106: Float = 2.12300514481811942347288949897e1;
const A107: Float = 1.52792336328824235832596922938e1;
const A108: Float = -3.32882109689848629194453265587e1;
const A109: Float = -2.03312017085086261358222928593e-2;

const A111: Float = -9.3714243008598732571704021658e-1;
const A114: Float = 5.18637242884406370830023853209e0;
const A115: Float = 1.09143734899672957818500254654e0;
const A116: Float = -8.14978701074692612513997267357e0;
const A117: Float = -1.85200656599969598641566180701e1;
const A118: Float = 2.27394870993505042818970056734e1;
const A119: Float = 2.49360555267965238987089396762e0;
const A1110: Float = -3.0467644718982195003823669022e0;

const A121: Float = 2.27331014751653820792359768449e0;
const A124: Float = -1.05344954667372501984066689879e1;
const A125: Float = -2.00087205822486249909675718444e0;
const A126: Float = -1.79589318631187989172765950534e1;
const A127: Float = 2.79488845294199600508499808837e1;
const A128: Float = -2.85899827713502369474065508674e0;
const A129: Float = -8.87285693353062954433549289258e0;
const A1210: Float = 1.23605671757943030647266201528e1;
const A1211: Float = 6.43392746015763530355970484046e-1;

const B1: Float = 5.42937341165687622380535766363e-2;
const B6: Float = 4.45031289275240888144113950566e0;
const B7: Float = 1.89151789931450038304281599044e0;
const B8: Float = -5.8012039600105847814672114227e0;
const B9: Float = 3.1116436695781989440891606237e-1;
const B10: Float = -1.52160949662516078556178806805e-1;
const B11: Float = 2.01365400804030348374776537501e-1;
const B12: Float = 4.47106157277725905176885569043e-2;

const BH1: Float = 0.244094488188976377952755905512e+00;
const BH2: Float = 0.733846688281611857341361741547e+00;
const BH3: Float = 0.220588235294117647058823529412e-01;

const ER1: Float = 0.1312004499419488073250102996e-01;
const ER6: Float = -0.1225156446376204440720569753e+01;
const ER7: Float = -0.4957589496572501915214079952e+00;
const ER8: Float = 0.1664377182454986536961530415e+01;
const ER9: Float = -0.3503288487499736816886487290e+00;
const ER10: Float = 0.3341791187130174790297318841e+00;
const ER11: Float = 0.8192320648511571246570742613e-01;
const ER12: Float = -0.2235530786388629525884427845e-01;

const A141: Float = 5.61675022830479523392909219681e-2;
const A147: Float = 2.53500210216624811088794765333e-1;
const A148: Float = -2.46239037470802489917441475441e-1;
const A149: Float = -1.24191423263816360469010140626e-1;
const A1410: Float = 1.5329179827876569731206322685e-1;
const A1411: Float = 8.20105229563468988491666602057e-3;
const A1412: Float = 7.56789766054569976138603589584e-3;
const A1413: Float = -8.298e-3;

const A151: Float = 3.18346481635021405060768473261e-2;
const A156: Float = 2.83009096723667755288322961402e-2;
const A157: Float = 5.35419883074385676223797384372e-2;
const A158: Float = -5.49237485713909884646569340306e-2;
const A1511: Float = -1.08347328697249322858509316994e-4;
const A1512: Float = 3.82571090835658412954920192323e-4;
const A1513: Float = -3.40465008687404560802977114492e-4;
const A1514: Float = 1.41312443674632500278074618366e-1;

const A161: Float = -4.28896301583791923408573538692e-1;
const A166: Float = -4.69762141536116384314449447206e0;
const A167: Float = 7.68342119606259904184240953878e0;
const A168: Float = 4.06898981839711007970213554331e0;
const A169: Float = 3.56727187455281109270669543021e-1;
const A1613: Float = -1.39902416515901462129418009734e-3;
const A1614: Float = 2.9475147891527723389556272149e0;
const A1615: Float = -9.15095847217987001081870187138e0;

const D41: Float = -0.84289382761090128651353491142e+01;
const D46: Float = 0.56671495351937776962531783590e+00;
const D47: Float = -0.30689499459498916912797304727e+01;
const D48: Float = 0.23846676565120698287728149680e+01;
const D49: Float = 0.21170345824450282767155149946e+01;
const D410: Float = -0.87139158377797299206789907490e+00;
const D411: Float = 0.22404374302607882758541771650e+01;
const D412: Float = 0.63157877876946881815570249290e+00;
const D413: Float = -0.88990336451333310820698117400e-01;
const D414: Float = 0.18148505520854727256656404962e+02;
const D415: Float = -0.91946323924783554000451984436e+01;
const D416: Float = -0.44360363875948939664310572000e+01;

const D51: Float = 0.10427508642579134603413151009e+02;
const D56: Float = 0.24228349177525818288430175319e+03;
const D57: Float = 0.16520045171727028198505394887e+03;
const D58: Float = -0.37454675472269020279518312152e+03;
const D59: Float = -0.22113666853125306036270938578e+02;
const D510: Float = 0.77334326684722638389603898808e+01;
const D511: Float = -0.30674084731089398182061213626e+02;
const D512: Float = -0.93321305264302278729567221706e+01;
const D513: Float = 0.15697238121770843886131091075e+02;
const D514: Float = -0.31139403219565177677282850411e+02;
const D515: Float = -0.93529243588444783865713862664e+01;
const D516: Float = 0.35816841486394083752465898540e+02;

const D61: Float = 0.19985053242002433820987653617e+02;
const D66: Float = -0.38703730874935176555105901742e+03;
const D67: Float = -0.18917813819516756882830838328e+03;
const D68: Float = 0.52780815920542364900561016686e+03;
const D69: Float = -0.11573902539959630126141871134e+02;
const D610: Float = 0.68812326946963000169666922661e+01;
const D611: Float = -0.10006050966910838403183860980e+01;
const D612: Float = 0.77771377980534432092869265740e+00;
const D613: Float = -0.27782057523535084065932004339e+01;
const D614: Float = -0.60196695231264120758267380846e+02;
const D615: Float = 0.84320405506677161018159903784e+02;
const D616: Float = 0.11992291136182789328035130030e+02;

const D71: Float = -0.25693933462703749003312586129e+02;
const D76: Float = -0.15418974869023643374053993627e+03;
const D77: Float = -0.23152937917604549567536039109e+03;
const D78: Float = 0.35763911791061412378285349910e+03;
const D79: Float = 0.93405324183624310003907691704e+02;
const D710: Float = -0.37458323136451633156875139351e+02;
const D711: Float = 0.10409964950896230045147246184e+03;
const D712: Float = 0.29840293426660503123344363579e+02;
const D713: Float = -0.43533456590011143754432175058e+02;
const D714: Float = 0.96324553959188282948394950600e+02;
const D715: Float = -0.39177261675615439165231486172e+02;
const D716: Float = -0.14972683625798562581422125276e+03;
