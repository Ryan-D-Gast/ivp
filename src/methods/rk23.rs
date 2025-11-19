//! Bogacki–Shampine 3(2) pair (RK23) adaptive-step integrator

use crate::{
    Float,
    error::{Error, ConfigError},
    interpolate::Interpolate,
    methods::{Evals, IntegrationResult, Steps, Tolerance, hinit},
    ivp::IVP,
    solout::{ControlFlag, SolOut},
    status::Status,
};
use bon::Builder;

/// Bogacki–Shampine 3(2) pair (RK23) adaptive-step integrator.
#[derive(Builder, Clone, Debug)]
pub struct RK23 {
    /// Safety factor for step size selection (default: 0.9)
    #[builder(default = 0.9)]
    pub safety_factor: Float,
    /// Minimum allowed scaling factor (default: 0.2)
    #[builder(default = 0.2)]
    pub scale_min: Float,
    /// Maximum allowed scaling factor (default: 10.0)
    #[builder(default = 10.0)]
    pub scale_max: Float,
    /// Maximum step size (default: None = unlimited)
    pub max_step: Option<Float>,
    /// Initial step size (default: None = auto)
    pub first_step: Option<Float>,
    /// Maximum number of steps (default: 10_000)
    #[builder(default = 10_000)]
    pub max_steps: usize,
    /// Enable dense output (default: true)
    #[builder(default = true)]
    pub dense_output: bool,
}

impl Default for RK23 {
    fn default() -> Self {
        Self {
            safety_factor: 0.9,
            scale_min: 0.2,
            scale_max: 10.0,
            max_step: None,
            first_step: None,
            max_steps: 10_000,
            dense_output: true,
        }
    }
}

impl RK23 {
    /// Bogacki–Shampine 3(2) pair (RK23) — adaptive solver with optional dense output.
    ///
    /// This function integrates the autonomous system `y' = f(xc, y)` from `x` to
    /// `xend`, advancing the provided state buffer `y` in-place. It performs
    /// classical error control using the embedded 2nd‑order estimate and can, if
    /// requested, provide dense-output coefficients for interpolation inside each
    /// step and call a user-provided `SolOut` hook.
    ///
    /// # Arguments
    ///
    /// ## Defining the Problem
    /// - `f`: Right‑hand side implementing `IVP`.
    /// - `x`: Initial independent variable value.
    /// - `xend`: Final independent variable value.
    /// - `y`: Mutable slice containing the initial state; on success contains the
    ///   state at the final time.
    /// - `rtol`, `atol`: Relative and absolute tolerances (see [`Tolerance`]).
    /// 
    /// ## Output Control
    /// - `solout`: Optional mutable reference to a `SolOut` callback used for
    ///   intermediate output. If `dense_output` is `true` the callback may receive
    ///   a dense interpolant.
    /// - `dense_output`: If `true`, dense‑output coefficients are computed every
    ///   accepted step to enable fast interpolation via the provided interpolant.
    /// 
    /// Solver settings (`safety_factor`, `scale_min`, `scale_max`, `max_step`, `first_step`, `max_steps`)
    /// are configured via the `RK23` struct fields.
    ///
    /// # Returns
    /// A `Result` with `IntegrationResult` on success or an `Error` if validation fails.
    pub fn solve<F, S>(
        &self,
        f: &F,
        x: &mut Float,
        y: &mut [Float],
        xend: Float,
        rtol: Tolerance,
        atol: Tolerance,
        mut solout: Option<&mut S>,
    ) -> Result<IntegrationResult, Error>
    where
        F: IVP,
        S: SolOut,
    {
        // --- Input Validation ---
        
        // Maximum Number of Steps
        let nmax = self.max_steps;
        if nmax == 0 {
            return Err(Error::Config(ConfigError::MustBePositive {
                parameter: "max_steps",
                value: nmax,
            }));
        }

        // Safety Factor
        let safety_factor = self.safety_factor;
        if safety_factor >= 1.0 || safety_factor <= 1e-4 {
            return Err(Error::Config(ConfigError::OutOfRange {
                parameter: "safety_factor",
                value: safety_factor,
                min: 1e-4,
                max: 1.0,
            }));
        }

        // Step size scaling factors
        let scale_min = self.scale_min;
        let scale_max = self.scale_max;
        if scale_min <= 0.0 || scale_max <= scale_min {
            return Err(Error::Config(ConfigError::InvalidScaleFactors {
                min: scale_min,
                max: scale_max,
            }));
        }

        // Error exponent
        let error_exponent = -1.0 / 3.0;

        // Maximum step size
        let hmax = self.max_step.map(|h| h.abs()).unwrap_or((xend - *x).abs());

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
        let mut xold = *x;
        let mut xout: Option<Float> = None;
        let posneg = (xend - *x).signum();

        // --- Initializations ---
        f.ode(*x, &y, &mut k1);
        evals.ode += 1;
        let mut h = match self.first_step {
            Some(h0) => h0.abs() * posneg,
            None => {
                evals.ode += 1;
                hinit(
                    f, *x, &y, posneg, &k1, &mut k2, &mut k3, 3, hmax, &atol, &rtol,
                )
            }
        };
        // Prepare a persistent interpolator object (pointer-based)
        let interpolator = &DenseOutput::new(
            cont.as_ptr(),
            cont.len(),
            &xold as *const Float,
            &h as *const Float,
        );

        // Initial SolOut call (no interpolator yet; xold == *x)
        if let Some(sol) = solout.as_mut() {
            match sol.solout::<DenseOutput>(xold, x, y, None) {
                ControlFlag::Interrupt => {
                    return Ok(IntegrationResult {
                        h,
                        status: Status::UserInterrupt,
                        evals,
                        steps,
                    });
                }
                ControlFlag::ModifiedSolution => {
                    // Recompute k1 at new (x, y).
                    f.ode(*x, &y, &mut k1);
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
            if steps.total >= nmax {
                status = Status::NeedLargerNMax;
                break;
            }

            // Check for last step adjustment
            if (*x + h - xend) * posneg > 0.0 {
                h = xend - *x;
            }

            // Stage 2
            for i in 0..n {
                yt[i] = y[i] + h * A21 * k1[i];
            }
            f.ode(*x + C2 * h, &yt, &mut k2);

            // Stage 3
            for i in 0..n {
                yt[i] = y[i] + h * A32 * k2[i];
            }
            f.ode(*x + C3 * h, &yt, &mut k3);

            // Compute solution and error estimate
            for i in 0..n {
                yt[i] = y[i] + h * (B1 * k1[i] + B2 * k2[i] + B3 * k3[i]);
            }

            // Stage 4/1: derivative at new point, also used as k1 if accepted.
            f.ode(*x + h, &yt, &mut k4);

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
                xold = *x;
                *x += h;

                // Prepare dense output
                if self.dense_output && solout.is_some() {
                    cont[0..n].copy_from_slice(&ye);
                    for i in 0..n {
                        cont[n + i] = k1[i];
                        cont[2 * n + i] = D21 * k1[i] + D22 * k2[i] + D23 * k3[i] + D24 * k4[i];
                        cont[3 * n + i] = D31 * k1[i] + D32 * k2[i] + D33 * k3[i] + D34 * k4[i];
                    }
                }

                // Optional callback function
                if let Some(sol) = solout.as_mut() {
                    let event = xout.map_or(false, |xo| xo <= *x);
                    let interpolation = if self.dense_output || event {
                        Some(interpolator)
                    } else {
                        None
                    };
                    match sol.solout(xold, x, y, interpolation) {
                        ControlFlag::Interrupt => {
                            status = Status::UserInterrupt;
                            break;
                        }
                        ControlFlag::ModifiedSolution => {
                            // Update with modified solution
                            // Recompute k1 at new (*x, y).
                            f.ode(*x, &y, &mut k1);
                            evals.ode += 1;
                        }
                        ControlFlag::XOut(xo) => {
                            xout = Some(xo);
                            // Reuse k4 as k1 for the next step to save an evaluation.
                            k1.copy_from_slice(&k4);
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
                if *x == xend {
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

        Ok(IntegrationResult::new(h, status, evals, steps))
    }

    /// Dense output evaluation for RK23
    pub fn interpolate(xi: Float, yi: &mut [Float], cont: &[Float], xold: Float, h: Float) {
        let n = yi.len();
        let xc = (xi - xold) / h;
        let x2 = xc * xc;
        let x3 = x2 * xc;
        for i in 0..n {
            yi[i] = cont[i] + h * (cont[n + i] * xc + cont[2 * n + i] * x2 + cont[3 * n + i] * x3);
        }
    }
}

// Dense output interpolator for RK23 (pointer-based like DOPRI style)
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
    fn interpolate(&self, ti: Float, yi: &mut [Float]) {
        unsafe {
            let cont = std::slice::from_raw_parts(self.cont_ptr, self.cont_len);
            let xold = *self.xold_ptr;
            let h = *self.h_ptr;
            RK23::interpolate(ti, yi, cont, xold, h);
        }
    }

    fn get_cont(&self) -> (Vec<Float>, Float, Float) {
        unsafe {
            let cont = std::slice::from_raw_parts(self.cont_ptr, self.cont_len);
            let xold = *self.xold_ptr;
            let h = *self.h_ptr;
            (cont.to_vec(), xold, h)
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

