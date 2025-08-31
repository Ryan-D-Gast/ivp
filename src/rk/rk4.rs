use crate::{Float, ODE, SolOut, ControlFlag};
use crate::interpolate::CubicHermite;
use crate::rk::{RKSettings, RKResult};
use crate::status::Status;
use crate::error::Error;

/// Classical explicit Runge-Kutta 4 (RK4) fixed-step integrator.
/// This implementation takes a single fixed step of size `h` from `x` to `x+h`.
/// It follows the library interface and provides a simple interpolator (cubic)
/// for the `SolOut` callback.
pub fn rk4<F, S>(
    f: &F,
    x: Float,
    y: &[Float],
    xend: Float,
    h: Float,
    solout: &mut S,
    settings: RKSettings,
) -> Result<RKResult, Error>
where
    F: ODE,
    S: SolOut,
{
    if h == 0.0 {
        return Err(Error::InvalidStepSize(h));
    }

    let direction = (xend - x).signum();
    if h.signum() != direction {
        return Err(Error::InvalidStepSize(h));
    }

    let result = rk4_core(f, x, y.to_vec(), xend, h, solout, settings.nmax);

    Ok(result)
}

fn rk4_core<F, S>(
    f: &F,
    mut x: Float,
    mut y: Vec<Float>,
    xend: Float,
    h: Float,
    solout: &mut S,
    nmax: usize,
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
    let mut nfcn = 0;
    let mut nstep = 0;
    let mut status = Status::Success;
    let mut xold = x;

    // Initial derivative at the starting point
    f.ode(x, &y, &mut k1);

    // Initial SolOut call
    let interp = CubicHermite::new(&xold, &h, &yt, &y, &k4, &k1);
    solout.solout(xold, x, &y, &interp);

    loop {
        if nstep >= nmax {
            status = Status::NeedLargerNmax;
            break;
        }

        // adjust last step so we land exactly on _xend
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
        for i in 0..n {
            yt[i] = y[i];
        }

        // Update state
        x += h;
        for i in 0..n {
            y[i] = y[i] + h * (B1 * k1[i] + B2 * k2[i] + B3 * k3[i] + B4 * k4[i]);
        }
        f.ode(x, &y, &mut k1);

        nfcn += 4;
        nstep += 1;

        // Interpolator
        let interp = CubicHermite::new(&xold, &h, &yt, &y, &k4, &k1);

        match solout.solout(xold, x, &y, &interp) {
            ControlFlag::Interrupt => {
                status = Status::Interrupted;
                break;
            }
            ControlFlag::ModifiedSolution => {
                // Recompute derivative
                f.ode(x + h, &y, &mut k1);
                nfcn += 1;
            }
            ControlFlag::Continue => {}
        }

        if last {
            break;
        }
    }

    RKResult {
        x,
        y: y.to_vec(),
        h,
        nfcn,
        nstep,
        status,
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
