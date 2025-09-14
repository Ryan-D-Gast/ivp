//! DOPRI5 - Dormand–Prince 8(5,3) explicit Runge–Kutta integrator
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
    methods::{
        hinit::hinit,
        result::IntegrationResult,
        settings::{Settings, Tolerance},
    },
    ode::ODE,
    solout::{ControlFlag, SolOut},
    status::Status,
};

/// Explicit Runge-Kutta method of order 5(4) due to
/// Dormand & Prince (with stepsize control and dense output).
pub fn dopri5<F, S>(
    f: &F,
    mut x: Float,
    xend: Float,
    y: &[Float],
    rtol: Tolerance,
    atol: Tolerance,
    mut solout: Option<&mut S>,
    settings: Settings,
) -> Result<IntegrationResult, Vec<Error>>
where
    F: ODE,
    S: SolOut,
{
    // --- Input Validation ---
    let mut errors: Vec<Error> = Vec::new();

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

    // Number of steps before performing a stiffness test
    let nstiff = match settings.nstiff {
        Some(n) => {
            if n <= 0 {
                errors.push(Error::NStiffMustBePositive(n));
            }
            n
        }
        None => 1000,
    };

    // Rounding Unit
    let uround = match settings.uround {
        Some(u) => {
            if u <= 1e-35 || u >= 1.0 {
                errors.push(Error::URoundOutOfRange(u));
            }
            u
        }
        None => 2.3e-16,
    };

    // Safety Factor
    let safety_factor = match settings.safety_factor {
        Some(f) => {
            if f >= 1.0 || f <= 1e-4 {
                errors.push(Error::SafetyFactorOutOfRange(f));
            }
            f
        }
        None => 0.9,
    };

    // Parameters for step size selection
    let facc1 = match settings.scale_min {
        Some(f) => 1.0 / f,
        None => 5.0,
    };
    let facc2 = match settings.scale_max {
        Some(f) => 1.0 / f,
        None => 1.0 / 10.0,
    };

    // Beta for step control stabilization
    let beta = match settings.beta {
        Some(b) => {
            if b > 0.2 {
                errors.push(Error::BetaTooLarge(b));
            }
            b
        }
        None => 0.04,
    };

    // Maximum step size
    let hmax = match settings.hmax {
        Some(h) => h.abs(),
        None => (xend - x).abs(),
    };

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
    let mut k5 = vec![0.0; n];
    let mut k6 = vec![0.0; n];
    let mut y1 = vec![0.0; n];
    let mut cont = vec![0.0; n * 5];
    let mut facold: Float = 1e-4;
    let mut last = false;
    let mut reject = false;
    let mut nonsti = 0;
    let mut hlamb = 0.0;
    let mut iasti = 0;
    let mut fac11;
    let mut fac;
    let mut hnew;
    let mut xph;
    let mut nfev: usize = 0;
    let mut nstep: usize = 0;
    let mut naccpt: usize = 0;
    let mut nrejct: usize = 0;
    let mut xold;
    let status;
    let expo1 = 0.2 - beta * 0.75;
    let posneg = (xend - x).signum();

    // --- Initializations ---
    f.ode(x, &y, &mut k1);
    nfev += 1;
    let mut h = match settings.h0 {
        Some(h0) => h0,
        None => {
            nfev += 1;
            hinit(
                f, x, &y, posneg, &k1, &mut k2, &mut y1, 5, hmax, &atol, &rtol,
            )
        }
    };
    if let Some(s) = solout.as_mut() {
        let interp = DenseOutput::new(&cont, x, h);
        s.solout(x, x, &y, &interp);
    }

    // --- Main integration loop ---
    loop {
        // Check for maximum number of steps
        if nstep > nmax {
            status = Status::NeedLargerNmax;
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

        nstep += 1;

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
        nfev += 6;

        // Prepare last segment of dense output before recalculating k4
        if solout.is_some() {
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
            naccpt += 1;

            // Stiffness detection
            if (naccpt % nstiff == 0) || (iasti > 0) {
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
                    nonsti = 0;
                    iasti += 1;
                    if iasti == 15 {
                        status = Status::ProbablyStiff;
                        break;
                    }
                } else {
                    nonsti += 1;
                    if nonsti == 6 {
                        iasti = 0;
                    }
                }
            }

            // Prepare dense output
            if solout.is_some() {
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

            // Optional callback function
            if let Some(ref mut s) = solout {
                match s.solout(xold, x, &y, &DenseOutput::new(&cont, xold, h)) {
                    ControlFlag::Interrupt => {
                        status = Status::Interrupted;
                        break;
                    }
                    ControlFlag::ModifiedSolution(xm, ym) => {
                        // Update with modified solution
                        x = xm;
                        y.copy_from_slice(&ym);

                        // Recompute k2 at new (x, y)
                        f.ode(x, &y, &mut k2);
                        nfev += 1;
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
            if hnew.abs() > hmax.abs() {
                hnew = posneg * hmax.abs();
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
            if naccpt > 1 {
                nrejct += 1;
            }
            last = false;
        }
        h = hnew;
    }

    Ok(IntegrationResult {
        x,
        y: y.to_vec(),
        h,
        status,
        nfev,
        njev: 0,
        nsol: 0,
        ndec: 0,
        nstep,
        naccpt,
        nrejct,
    })
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
struct DenseOutput<'a> {
    cont: &'a [Float],
    xold: Float,
    h: Float,
}

impl<'a> DenseOutput<'a> {
    fn new(cont: &'a [Float], xold: Float, h: Float) -> Self {
        Self { cont, xold, h }
    }
}

impl<'a> Interpolate for DenseOutput<'a> {
    fn interpolate(&self, xi: Float, yi: &mut [Float]) {
        contdp5(xi, yi, self.cont, self.xold, self.h);
    }

    fn get_cont(&self) -> (Vec<Float>, Float, Float) {
        (self.cont.to_vec(), self.xold, self.h)
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
