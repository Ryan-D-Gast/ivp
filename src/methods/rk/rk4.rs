//! Classic explicit Runge-Kutta 4 (RK4) fixed-step integrator.

use crate::{
    Float,
    error::Error,
    interpolate::Interpolate,
    methods::{
        result::{IntegrationResult, Evals, Steps},
        settings::Settings
    },
    ode::ODE,
    solout::{ControlFlag, SolOut},
    status::Status,
};

/// Classical explicit Runge-Kutta 4 (RK4) fixed-step integrator.
/// Provides a dense output via cubic Hermite interpolation.
pub fn rk4<F, S>(
    f: &F,
    mut x: Float,
    xend: Float,
    y: &mut [Float],
    h: Float,
    mut solout: Option<&mut S>,
    settings: Settings,
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
    let nmax = match settings.nmax {
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

    // --- Initializations ---
    f.ode(x, &y, &mut k1);
    if let Some(s) = solout.as_mut() {
        cont[0..n].copy_from_slice(&y);
        for i in 0..n {
            cont[n + i] = k1[i];
        }
        let interp = DenseOutput::new(&cont, xold, h);
        s.solout(xold, x, &y, &interp);
    }

    // --- Main integration loop ---
    loop {
        // Check for maximum number of steps
        if steps.total >= nmax {
            status = Status::NeedLargerNmax;
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

        // Prepare dense output
        if solout.is_some() {
            cont[0..n].copy_from_slice(&yt);
            for i in 0..n {
                cont[n + i] = k4[i];
                cont[2 * n + i] = k1[i];
            }
            cont[3 * n..4 * n].copy_from_slice(&y);
        }

        // Optional callback function
        if let Some(s) = solout.as_mut() {
            match s.solout(xold, x, &y, &DenseOutput::new(&cont, xold, h)) {
                ControlFlag::Interrupt => {
                    status = Status::Interrupted;
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

struct DenseOutput<'a> {
    cont: &'a [Float],
    xold: Float,
    h: Float,
}

impl<'a> DenseOutput<'a> {
    pub fn new(cont: &'a [Float], xold: Float, h: Float) -> Self {
        Self { xold, h, cont }
    }
}

impl<'a> Interpolate for DenseOutput<'a> {
    fn interpolate(&self, xi: Float, yi: &mut [Float]) {
        contrk4(xi, yi, self.cont, self.xold, self.h);
    }

    fn get_cont(&self) -> (Vec<Float>, Float, Float) {
        (self.cont.to_vec(), self.xold, self.h)
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
