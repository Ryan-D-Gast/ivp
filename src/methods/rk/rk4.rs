//! Classic explicit Runge–Kutta 4 (RK4) fixed-step integrator

use crate::{
    Float,
    error::Error,
    interpolate::Interpolate,
    methods::common::{Evals, IntegrationResult, Steps},
    ode::ODE,
    solout::{ControlFlag, SolOut},
    status::Status,
};

/// Classical explicit Runge–Kutta 4 (RK4) — fixed-step solver with optional dense output.
///
/// This function integrates the autonomous system `y' = f(x, y)` from `x` to
/// `xend` using a constant step size `h`, advancing the state buffer `y`
/// in-place. It can optionally provide dense-output coefficients for continuous
/// interpolation inside each step and call a user-provided `SolOut` hook.
///
/// # Arguments
///
/// - `f`: Right‑hand side implementing `ODE`.
/// - `x`: Initial independent variable value.
/// - `xend`: Final independent variable value.
/// - `y`: Mutable slice for the initial state; on success contains the state at `xend`.
/// - `h`: Fixed step size (its sign must match `xend - x`).
/// - `solout`: Optional mutable reference to a `SolOut` callback invoked once
///   before the loop and after each accepted step.
/// - `dense_output`: If `true`, dense‑output coefficients are computed for each
///   accepted step and an interpolant is passed to the callback.
/// - `max_steps`: Optional upper bound on the number of steps (default `100_000`).
///
/// # Returns
/// A `Result` with `IntegrationResult` on success or a vector of `Error`
/// values describing input validation issues.
pub fn rk4<F, S>(
    f: &F,
    mut x: Float,
    xend: Float,
    y: &mut [Float],
    h: Float,
    mut solout: Option<&mut S>,
    dense_output: bool,
    max_steps: Option<usize>,
) -> Result<IntegrationResult, Vec<Error>>
where
    F: ODE,
    S: SolOut,
{
    // --- Input Validation ---
    let mut errors = Vec::new();

    // Initial Step Size
    let direction = (xend - x).signum();
    if h == 0.0 || h.signum() != direction {
        errors.push(Error::InvalidStepSize(h));
    }

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
    let mut cont = vec![0.0; 4 * n];
    let mut evals = Evals::new();
    let mut steps = Steps::new();
    let mut status = Status::Success;
    let mut xold = x;
    let mut xout: Option<Float> = None;

    // --- Initializations ---
    f.ode(x, &y, &mut k1);
    // Prepare a persistent interpolator object (pointer-based) used when needed
    let interpolator = &DenseOutput::new(
        cont.as_ptr(),
        cont.len(),
        &xold as *const Float,
        &h as *const Float,
    );

    // Initial SolOut call (no interpolator yet; xold == x)
    if let Some(sol) = solout.as_mut() {
        match sol.solout::<DenseOutput>(xold, x, &y, None) {
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
                for i in 0..n {
                    y[i] = ym[i];
                }
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

        // Adjust last step so we land exactly on _xend
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

        x += h;
        for i in 0..n {
            y[i] += h * (B1 * k1[i] + B2 * k2[i] + B3 * k3[i] + B4 * k4[i]);
        }
        f.ode(x, &y, &mut k1);

        evals.ode += 4;
        steps.total += 1;

        // Decide if we must build dense output (for user xout events as well)
        let event = xout.map_or(false, |xo| xo <= x);
        if (dense_output || event) && solout.is_some() {
            cont[0..n].copy_from_slice(&yt);
            for i in 0..n {
                cont[n + i] = k4[i];
                cont[2 * n + i] = k1[i];
            }
            cont[3 * n..4 * n].copy_from_slice(&y);
        }

        // Optional callback function
        if let Some(sol) = solout.as_mut() {
            let interpolation = if dense_output || xout.map_or(false, |xo| xo <= x) {
                Some(interpolator)
            } else {
                None
            };
            match sol.solout(xold, x, &y, interpolation) {
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

    Ok(IntegrationResult::new(x, h, status, evals, steps))
}

/// Continuous output function for RK4 using cubic Hermite interpolation.
pub fn contrk4(xi: Float, yi: &mut [Float], cont: &[Float], xold: Float, h: Float) {
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

/// Dense output interpolator for RK4 (pointer-based like DOPRI/DOP853 style)
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
            let cont = std::slice::from_raw_parts(self.cont_ptr, self.cont_len);
            let xold = *self.xold_ptr;
            let h = *self.h_ptr;
            contrk4(xi, yi, cont, xold, h);
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
