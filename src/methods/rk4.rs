//! Classic explicit Runge–Kutta 4 (RK4) fixed-step integrator

use crate::{
    Float,
    dense::StepInterpolant,
    error::{Error, ConfigError},
    methods::{Evals, IntegrationResult, Steps},
    ivp::IVP,
    solout::{ControlFlag, SolOut},
    status::Status,
};
use bon::Builder;

/// Classic explicit Runge–Kutta 4 (RK4) fixed-step integrator with configuration.
#[derive(Builder, Clone, Debug)]
pub struct RK4 {
    /// Maximum number of steps (default: 100_000)
    #[builder(default = 100_000)]
    pub max_steps: usize,
    /// Enable dense output (default: true)
    #[builder(default = true)]
    pub dense_output: bool,
}

impl Default for RK4 {
    fn default() -> Self {
        Self {
            max_steps: 100_000,
            dense_output: true,
        }
    }
}

impl RK4 { 
    /// Classical explicit Runge–Kutta 4 (RK4) — fixed-step solver with optional dense output.
    ///
    /// This function integrates the autonomous system `y' = f(x, y)` from `x` to
    /// `xend` using a constant step size `h`, advancing the state buffer `y`
    /// in-place. It can optionally provide dense-output coefficients for continuous
    /// interpolation inside each step and call a user-provided `SolOut` hook.
    ///
    /// # Arguments
    ///
    /// ## Defining the Problem
    /// - `f`: Right‑hand side implementing `IVP`.
    /// - `x`: Initial independent variable value.
    /// - `xend`: Final independent variable value.
    /// - `y`: Mutable slice for the initial state; on success contains the state at `xend`.
    /// - `h`: Fixed step size (its sign must match `xend - x`).
    /// 
    /// ## Output Control
    /// - `solout`: Optional mutable reference to a `SolOut` callback invoked once
    ///   before the loop and after each accepted step.
    /// - `dense_output`: If `true`, dense‑output coefficients are computed for each
    ///   accepted step and an interpolant is passed to the callback.
    /// 
    /// ## Optional Settings
    /// - `max_steps`: Optional upper bound on the number of steps (default `100_000`).
    ///
    /// Solver settings are configured via the `RK4` struct fields.
    ///
    /// # Returns
    /// A `Result` with `IntegrationResult` on success or an `Error` if validation fails.
    pub fn solve<F, S>(
        &self,
        f: &F,
        x0: Float,
        y0: &[Float],
        xend: Float,
        h: Float,
        mut solout: Option<&mut S>,
    ) -> Result<IntegrationResult, Error>
    where
        F: IVP,
        S: SolOut,
    {
        // Create mutable copies for the solver to mutate
        let mut x = x0;
        let mut y = y0.to_vec();

        // --- Input Validation ---
        
        // Initial Step Size
        let posneg = (xend - x).signum();
        if h == 0.0 || h.signum() != posneg {
            return Err(Error::Config(ConfigError::InvalidStepSize {
                value: h,
                expected_sign: posneg,
            }));
        }

        // Maximum Number of Steps
        let nmax = self.max_steps;
        if nmax == 0 {
            return Err(Error::Config(ConfigError::MustBePositive {
                parameter: "max_steps",
                value: nmax,
            }));
        }

        // --- Declarations ---
        let n = y.len();
        let mut k1 = vec![0.0; n];
        let mut k2 = vec![0.0; n];
        let mut k3 = vec![0.0; n];
        let mut k4 = vec![0.0; n];
        let mut yt = vec![0.0; n];
        let mut cont = vec![0.0; 4 * n];
        let mut evals = Evals::new();
        let mut steps = Steps::new();
        let mut status = Status::Success;
        let mut xold = x;
        let mut xout: Option<Float> = None;

        // --- Initializations ---
        f.ode(x, &y, &mut k1);

        // Initial SolOut call (no interpolator yet; xold == x)
        if let Some(sol) = solout.as_mut() {
            match sol.solout(xold, &mut x, &mut y, None) {
                ControlFlag::Interrupt => {
                    return Ok(IntegrationResult {
                        h,
                        status: Status::UserInterrupt,
                        evals,
                        steps,
                    });
                }
                ControlFlag::ModifiedSolution => {
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
            if steps.total >= nmax {
                status = Status::NeedLargerNMax;
                break;
            }

            // Adjust last step so we land exactly on xend
            let mut last = false;
            if (x + 1.01 * h - xend) * h.signum() > 0.0 {
                last = true;
            }

            // Stage computations
            for i in 0..n {
                yt[i] = y[i] + h * A21 * k1[i];
            }
            f.ode(x + C2 * h, &yt, &mut k2);

            for i in 0..n {
                yt[i] = y[i] + h * A32 * k2[i];
            }
            f.ode(x + C3 * h, &yt, &mut k3);

            for i in 0..n {
                yt[i] = y[i] + h * A43 * k3[i];
            }
            f.ode(x + C4 * h, &yt, &mut k4);

            xold = x;
            yt.copy_from_slice(&y);

            // Update solution
            x += h;
            for i in 0..n {
                y[i] += h * (B1 * k1[i] + B2 * k2[i] + B3 * k3[i] + B4 * k4[i]);
            }
            f.ode(x, &y, &mut k1);

            evals.ode += 4;
            steps.total += 1;

            // Decide if we must build dense output (for user xout events as well)
            let event = xout.map_or(false, |xo| xo <= x);
            if (self.dense_output || event) && solout.is_some() {
                cont[0..n].copy_from_slice(&yt);
                for i in 0..n {
                    cont[n + i] = k4[i];
                    cont[2 * n + i] = k1[i];
                }
                cont[3 * n..4 * n].copy_from_slice(&y);
            }

            // Optional callback function
            if let Some(sol) = solout.as_mut() {
                let interpolant = if self.dense_output || xout.map_or(false, |xo| xo <= x) {
                    Some(StepInterpolant::new(&cont, xold, h, Self::interpolate))
                } else {
                    None
                };
                match sol.solout(xold, &mut x, &mut y, interpolant.as_ref()) {
                    ControlFlag::Interrupt => {
                        status = Status::UserInterrupt;
                        break;
                    }
                    ControlFlag::ModifiedSolution => {
                        // Recompute k1 at new (x, y).
                        f.ode(x, &y, &mut k1);
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
                break;
            }
        }

        // Final update happens on accepted step, no need to update here again
        Ok(IntegrationResult::new(h, status, evals, steps))
    }

    /// Continuous output function for RK4 using cubic Hermite interpolation.
    pub fn interpolate(xi: Float, yi: &mut [Float], cont: &[Float], xold: Float, h: Float) {
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
                + h01 * cont[3 * n + i]
                + h11 * h * cont[2 * n + i];
        }
    }
}

// Classical RK4 coefficients
const C2: Float = 0.5;
const C3: Float = 0.5;
const C4: Float = 1.0;
const A21: Float = 0.5;
const A32: Float = 0.5;
const A43: Float = 1.0;
const B1: Float = 1.0 / 6.0;
const B2: Float = 1.0 / 3.0;
const B3: Float = 1.0 / 3.0;
const B4: Float = 1.0 / 6.0;
