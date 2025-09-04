//! Classic explicit Runge-Kutta 4 (RK4) fixed-step integrator.

use crate::{
    ControlFlag, Float, ODE, SolOut, error::Error, interpolate::CubicHermite, args::Args,
    solution::Solution, status::Status,
};

/// Classical explicit Runge-Kutta 4 (RK4) fixed-step integrator.
/// Provides a dense output via cubic Hermite interpolation.
pub fn rk4<F, S>(
    f: &F,
    mut x: Float,
    xend: Float,
    y: &[Float],
    h: Float,
    args: Args<S>,
) -> Result<Solution, Error>
where
    F: ODE,
    S: SolOut,
{
    // --- Input Validation ---

    // Callback function
    let mut solout = args.solout;

    if h == 0.0 {
        return Err(Error::InvalidStepSize(h));
    }

    let direction = (xend - x).signum();
    if h.signum() != direction {
        return Err(Error::InvalidStepSize(h));
    }

    // --- Declarations ---
    let n = y.len();
    let mut y = y.to_vec();
    let mut k1 = vec![0.0; n];
    let mut k2 = vec![0.0; n];
    let mut k3 = vec![0.0; n];
    let mut k4 = vec![0.0; n];
    let mut yt = vec![0.0; n];
    let mut nfev = 0;
    let mut nstep = 0;
    let mut status = Status::Success;
    let mut xold = x;
    let nmax = args.nmax;

    // --- Initializations ---
    f.ode(x, &y, &mut k1);
    if let Some(s) = solout.as_mut() {
        let interp = CubicHermite::new(&xold, &h, &yt, &y, &k4, &k1);
        s.solout(xold, x, &y, &interp);
    }

    // --- Main integration loop ---
    loop {
        // Check for maximum number of steps
        if nstep >= nmax {
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

        // Store previous state
        xold = x;
        yt.copy_from_slice(&y);

        // Update state
        x += h;
        for i in 0..n {
            y[i] = y[i] + h * (B1 * k1[i] + B2 * k2[i] + B3 * k3[i] + B4 * k4[i]);
        }
        f.ode(x, &y, &mut k1);

        nfev += 4;
        nstep += 1;

        // Optional Callback function
        match solout.as_mut().map_or(ControlFlag::Continue, |s| {
            let interp = CubicHermite::new(&xold, &h, &yt, &y, &k4, &k1);
            s.solout(xold, x, &y, &interp)
        }) {
            ControlFlag::Interrupt => {
                status = Status::Interrupted;
                break;
            }
            ControlFlag::ModifiedSolution => {
                // Recompute derivative
                f.ode(x + h, &y, &mut k1);
                nfev += 1;
            }
            ControlFlag::Continue => {}
        }

        if last {
            break;
        }
    }

    Ok(Solution {
        x,
        y: y.to_vec(),
        h,
        nfev,
        nstep,
        naccpt: nstep,
        nrejct: 0,
        status,
    })
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
