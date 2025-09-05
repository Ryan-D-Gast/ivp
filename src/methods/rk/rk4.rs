//! Classic explicit Runge-Kutta 4 (RK4) fixed-step integrator.

use crate::{
    Float,
    core::{
        interpolate::Interpolate,
        ode::ODE,
        solout::{ControlFlag, SolOut},
        solution::Solution,
        status::Status,
    },
    error::Error,
    methods::settings::Settings,
};

/// Classical explicit Runge-Kutta 4 (RK4) fixed-step integrator.
/// Provides a dense output via cubic Hermite interpolation.
pub fn rk4<F, S>(
    f: &F,
    mut x: Float,
    xend: Float,
    y: &[Float],
    h: Float,
    mut solout: Option<&mut S>,
    settings: Settings,
) -> Result<Solution, Vec<Error>>
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

    if !errors.is_empty() {
        return Err(errors);
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
    let nmax = settings.nmax;

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
            ControlFlag::ModifiedSolution(xm, ym) => {
                // Update with modified solution
                x = xm;
                y = ym;

                // Recompute k1 at new (x, y).
                f.ode(x, &y, &mut k1);
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

struct CubicHermite<'a> {
    x0: &'a Float,
    h: &'a Float,
    y0: &'a [Float],
    y1: &'a [Float],
    dy0: &'a [Float],
    dy1: &'a [Float],
}

impl<'a> CubicHermite<'a> {
    pub fn new(
        x0: &'a Float,
        h: &'a Float,
        y0: &'a [Float],
        y1: &'a [Float],
        dy0: &'a [Float],
        dy1: &'a [Float],
    ) -> Self {
        Self {
            x0,
            h,
            y0,
            y1,
            dy0,
            dy1,
        }
    }
}

impl<'a> Interpolate for CubicHermite<'a> {
    fn interpolate(&self, xi: Float, yi: &mut [Float]) {
        // Cubic Hermite interpolation
        let t = (xi - self.x0) / self.h;
        let t2 = t * t;
        let t3 = t2 * t;

        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        for i in 0..self.y0.len() {
            yi[i] = h00 * self.y0[i]
                + h10 * self.h * self.dy0[i]
                + h01 * self.y1[i]
                + h11 * self.h * self.dy1[i];
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
