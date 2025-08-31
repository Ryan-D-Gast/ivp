use crate::{Float, ODE, SolOut, Interpolate, ControlFlag};
use crate::rk::{RKSettings, RKResult, RKInputError};
use crate::status::Status;

/// Classical explicit Runge-Kutta 4 (RK4) fixed-step integrator.
/// This implementation takes a single fixed step of size `h` from `x` to `x+h`.
/// It follows the library interface and provides a simple interpolator (cubic)
/// for the `SolOut` callback.
pub fn rk4<const N: usize, F, S>(
    f: &mut F,
    x: Float,
    y: [Float; N],
    xend: Float,
    h: Float,
    solout: &mut S,
    settings: RKSettings<N>,
) -> Result<RKResult<N>, RKInputError>
where
    F: ODE<N>,
    S: SolOut<N>,
{
    if h == 0.0 {
        return Err(RKInputError::InvalidStepSize(h));
    }

    let direction = (xend - x).signum();
    if h.signum() != direction {
        return Err(RKInputError::InvalidStepSize(h));
    }

    let result = rk4_core::<N, F, S>(f, x, y, xend, h, solout, settings.nmax);

    Ok(result)
}

/// Core RK4 implementation scaffold. Replace the comments below with the actual
/// stage computations, error handling (if any), and calls to `solout`.
fn rk4_core<const N: usize, F, S>(
    f: &mut F,
    mut x: Float,
    mut y: [Float; N],
    xend: Float,
    h: Float,
    solout: &mut S,
    nmax: usize,
) -> RKResult<N>
where
    F: ODE<N>,
    S: SolOut<N>,
{
    // --- Local workspace ---
    let mut k1 = [0.0; N];
    let mut k2 = [0.0; N];
    let mut k3 = [0.0; N];
    let mut k4 = [0.0; N];
    let mut yt = [0.0; N];
    let mut nfcn = 0;
    let mut nstep = 0;
    let mut status = Status::Success;
    let mut xold = x;

    // Initial derivative at the starting point
    f.ode(x, &y, &mut k1);

    // Initial SolOut call
    let interp = RK4Interp {
        x0: &xold,
        h: &h,
        y0: &yt,
        y1: &y,
        dy0: &k4,
        dy1: &k1,
    };
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
        for i in 0..N {
            yt[i] = y[i] + h * A21 * k1[i];
        }
        f.ode(x + C2 * h, &yt, &mut k2);

        for i in 0..N {
            yt[i] = y[i] + h * A32 * k2[i];
        }
        f.ode(x + C3 * h, &yt, &mut k3);

        for i in 0..N {
            yt[i] = y[i] + h * A43 * k3[i];
        }
        f.ode(x + C4 * h, &yt, &mut k4);

        // Store previous state
        xold = x;
        yt = y;

        // Update state
        x += h;
        for i in 0..N {
            y[i] = y[i] + h * (B1 * k1[i] + B2 * k2[i] + B3 * k3[i] + B4 * k4[i]);
        }
        f.ode(x, &y, &mut k1);

        nfcn += 4;
        nstep += 1;

        // Interpolator uses derivative at start (k1) and at end (k4)
        let interp = RK4Interp {
            x0: &xold,
            h: &h,
            y0: &yt,
            y1: &y,
            dy0: &k4,
            dy1: &k1,
        };

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
        y,
        h,
        nfcn,
        nstep,
        status,
    }
}

struct RK4Interp<'a, const N: usize> {
    x0: &'a Float,
    h: &'a Float,
    y0: &'a [Float; N],
    y1: &'a [Float; N],
    dy0: &'a [Float; N],
    dy1: &'a [Float; N],
}

impl<'a, const N: usize> Interpolate<N> for RK4Interp<'a, N> {
    fn interpolate(&self, xi: Float) -> [Float; N] {
        // Cubic Hermite interpolation on [x0, x0+h]
        let t = (xi - self.x0) / self.h;
        let t2 = t * t;
        let t3 = t2 * t;

        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        let mut y = [0.0; N];
        for i in 0..N {
            y[i] = h00 * self.y0[i]
                + h10 * self.h * self.dy0[i]
                + h01 * self.y1[i]
                + h11 * self.h * self.dy1[i];
        }
        y
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
