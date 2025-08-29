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
//! # Contains
//! - [`dopri5`] - The main function for the DOPRI5 integrator
//! - [`ODE<N>`] - Trait for defining the ODE system
//! - [`SolOut<N>`] - Trait for handling solution output
//! - [`DPSettings`] - Struct for configuring the integrator
//! - [`DPResult`] - Struct for holding the integration result
//! - [`contd5`] - Function for dense output interpolation
//! 

use crate::{
    Float,
    solout::{ControlFlag, SolOut},
    tolerance::Tolerance,
    ode::ODE,
    hinit::hinit,
    settings::DPSettings,
    status::Status,
    result::DPResult
};

/// Numerical solution of a system of first order
/// ordinary differential equations in the form
/// `y' = f(x, y)`. This is an explicit Runge-Kutta
/// method of order 8(5,3) due to dormand & prince
/// (with stepsize control and dense output).
///
/// # Summary
/// - Implement the [`ODE<N>`] trait for your system y' = f(x, y).
/// - Provide an implementation of [`SolOut<N>`] if you want to receive
///   callbacks after each accepted step. The callback receives the dense-output
///   coefficients (`cont`) and the step size `h`, so you can interpolate
///   inside the step using [`contd5`].
/// - Settings can be adjusted using the [`DPSettings`] struct.
/// - Call [`dopri5`] to perform the integration; it returns a
///   [`DPResult`] containing the final solution and statistics.
///
/// # Example
/// ```ignore
/// use dopri5::{dopri5, ODE, SolOut, DPSettings, contd5, ControlFlag};
/// // Van der Pol system
/// struct VanDerPol { eps: f64 }
/// impl ODE<2> for VanDerPol {
///     fn ode(&mut self, _x: f64, y: &[f64;2], dydx: &mut [f64;2]) {
///         dydx[0] = y[1];
///         dydx[1] = ((1.0 - y[0].powi(2)) * y[1] - y[0]) / self.eps;
///     }
/// }
///
/// // Prints evenly spaced points
/// struct Printer { xout: f64, dx: f64 }
/// impl Printer { fn new() -> Self { Self { xout: 0.0, dx: 0.1 } } }
/// impl SolOut<2> for Printer {
///     fn solout(&mut self, nr: usize, xold: f64, x: f64, y: &[f64;2], cont: &[[f64;2];8], h: f64) -> ControlFlag {
///         if nr == 1 {
///             println!("x = {:>5.2}, y = {:?}", xold, y);
///             self.xout = xold + self.dx;
///         }
///         while self.xout <= x {
///             let mut yi = [0.0f64; 2];
///             contd5(cont, xold, h, self.xout, &mut yi);
///             println!("x = {:>5.2}, y = {:?}", self.xout, yi);
///             self.xout += self.dx;
///         }
///         ControlFlag::Continue
///     }
/// }
///
/// fn main() {
///     let mut vdp = VanDerPol { eps: 1e-3 };
///     let x0 = 0.0;
///     let xend = 2.0;
///     let y0 = [2.0f64, 0.0f64];
///     let settings = DPSettings::default();
///     let mut printer = Printer::new();
///     let rtol = 1e-9;
///     let atol = 1e-9;
///     let res = dopri5(&mut vdp, x0, y0, xend, rtol, atol, &mut printer, settings).unwrap();
///     println!("finished: {:?}", res.status);
/// }
/// ```
pub fn dopri5<const N: usize, F, S, R, A>(
    f: &mut F,
    x: Float,
    y: [Float; N],
    xend: Float,
    rtol: R,
    atol: A,
    solout: &mut S,
    settings: DPSettings<N>,
) -> Result<DPResult<N>, Vec<String>>
where
    F: ODE<N>,
    S: SolOut<N>,
    R: Into<Tolerance<N>>,
    A: Into<Tolerance<N>>,
{
    // --- Declarations ---
    let nfcns: usize = 0;
    let nstep: usize = 0;
    let naccpt: usize = 0;
    let nrejct: usize = 0;

    // --- Input Validation ---
    let mut errors = Vec::new();

    // Maximum Number of Steps
    let nmax = settings.nmax;
    if nmax <= 0 {
        errors.push("nmax must be positive".to_string());
    }

    // Parameter for stiffness detection
    let nstiff = settings.nstiff;
    if nstiff <= 0 {
        errors.push("nstiff must be positive".to_string());
    }

    // Rounding Unit
    let uround = settings.uround;
    if uround <= 1e-35 || uround >= 1.0 {
        errors.push("uround must be in (1e-35, 1.0)".to_string());
    }

    // Safety Factor
    let safety_factor = settings.safety_factor;
    if safety_factor >= 1.0 || safety_factor <= 1e-4 {
        errors.push("safety_factor must be in (1e-4, 1.0)".to_string());
    }

    // Parameters for step size selection
    let (mut fac1, mut fac2) = settings.fac;
    if fac1 == 0.0 {
        fac1 = 0.2;
    }
    if fac2 == 0.0 {
        fac2 = 10.0;
    }

    // Beta for step control stabilization
    let mut beta = settings.beta;
    if beta < 0.0 {
        beta = 0.0;
    }
    if beta > 0.2 {
        errors.push("beta must be <= 0.2".to_string());
    }

    // Maximum step size
    let hmax = match settings.h_max {
        Some(h) => h.abs(),
        None => (xend - x).abs(),
    };

    // Initial step size: when not provided, use 0.0 so the core solver calls hinit
    let h = settings.h0.unwrap_or(0.0);

    if !errors.is_empty() {
        return Err(errors);
    }

    // --- Call DOPRI5 Core Solver ---
    let rtol = rtol.into();
    let atol = atol.into();
    let result = dp5co::<N, F, S>(
        f,
        x,
        y,
        xend,
        hmax,
        h,
        rtol,
        atol,
        solout,
        nmax,
        uround,
        nstiff,
        safety_factor,
        beta,
        fac1,
        fac2,
        nfcns,
        nstep,
        naccpt,
        nrejct,
    );

    Ok(result)
}

/// DOPRI5 core solver
fn dp5co<const N: usize, F, S>(
    f: &mut F,
    mut x: Float,
    mut y: [Float; N],
    xend: Float,
    hmax: Float,
    mut h: Float,
    rtol: Tolerance<N>,
    atol: Tolerance<N>,
    solout: &mut S,
    nmax: usize,
    uround: Float,
    nstiff: usize,
    safety_factor: Float,
    beta: Float,
    fac1: Float,
    fac2: Float,
    mut nfcns: usize,
    mut nstep: usize,
    mut naccpt: usize,
    mut nrejct: usize,
) -> DPResult<N>
where
    F: ODE<N>,
    S: SolOut<N>,
{
    // --- Initializations ---
    let mut k1 = [0.0; N];
    let mut k2 = [0.0; N];
    let mut k3 = [0.0; N];
    let mut k4 = [0.0; N];
    let mut k5 = [0.0; N];
    let mut k6 = [0.0; N];
    let mut y1 = [0.0; N];
    let mut cont = [[0.0; N]; 5];
    let mut facold: Float = 1e-4;
    let mut last = false;
    let mut reject = false;
    let mut nonsti = 0;
    let mut hlamb = 0.0;
    let mut iasti = 0;
    let mut xold = x;
    let mut fac11;
    let mut fac;
    let mut hnew;
    let status;
    let expo1 = 0.2 - beta * 0.75;
    let facc1 = 1.0 / fac1;
    let facc2 = 1.0 / fac2;
    let posneg = (xend - x).signum();
    let iord = 5;

    // Initial call
    f.ode(x, &y, &mut k1);
    nfcns += 1;
    if h == 0.0 {
        h = hinit(f, x, &y, posneg, &k1, &mut k2, &mut y1, iord, hmax, &atol, &rtol);
        nfcns += 1;
    }
    solout.solout(naccpt + 1, xold, x, &y, &cont, h);

    // Main integration loop
    loop {
        // check for maximum number of steps
        if nstep > nmax {
            status = Status::NeedLargerNmax;
            break;
        }

        // check for underflow due to machine rounding
        if 0.1 * h.abs() <= x.abs() * uround {
            status = Status::StepSizeTooSmall;
            break;
        }

        // adjust last step to land on xend
        if (x + 1.01 * h - xend) * posneg > 0.0 {
            h = xend - x;
            last = true;
        }

        nstep += 1;

        // Stage 2
        for i in 0..N {
            y1[i] = y[i] + h * A21 * k1[i];
        }
        f.ode(x + C2 * h, &y1, &mut k2);

        // Stage 3
        for i in 0..N {
            y1[i] = y[i] + h * (A31 * k1[i] + A32 * k2[i]);
        }
        f.ode(x + C3 * h, &y1, &mut k3);

        // Stage 4
        for i in 0..N {
            y1[i] = y[i] + h * (A41 * k1[i] + A42 * k2[i] + A43 * k3[i]);
        }
        f.ode(x + C4 * h, &y1, &mut k4);

        // Stage 5
        for i in 0..N {
            y1[i] = y[i] + h * (A51 * k1[i] + A52 * k2[i] + A53 * k3[i] + A54 * k4[i]);
        }
        f.ode(x + C5 * h, &y1, &mut k5);

        // Stage 6 (ysti)
        for i in 0..N {
            y1[i] = y[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]);
        }
        let xph = x + h;
        f.ode(xph, &y1, &mut k6);

        // Final stage
        for i in 0..N {
            y1[i] = y[i] + h * (A71 * k1[i] + A73 * k3[i] + A74 * k4[i] + A75 * k5[i] + A76 * k6[i]);
        }
        f.ode(xph, &y1, &mut k2);
        nfcns += 6;

        // K4 scaled for error estimate
        for i in 0..N {
            k4[i] = (E1 * k1[i] + E3 * k3[i] + E4 * k4[i] + E5 * k5[i] + E6 * k6[i] + E7 * k2[i]) * h;
        }

        // Error estimation
        let mut err = 0.0_f64;
        let n = N as f64;
        for i in 0..N {
            let sk = atol[i] + rtol[i] * y[i].abs().max(y1[i].abs());
            err += (k4[i] / sk) * (k4[i] / sk);
        }
        err = (err / n).sqrt();

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
                for i in 0..N {
                    let d1 = k2[i] - k6[i];
                    let ysti = y[i] + h * (A61 * k1[i] + A62 * k2[i] + A63 * k3[i] + A64 * k4[i] + A65 * k5[i]);
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

            // Dense output coefficient computation
            for i in 0..N {
                let yd0 = y[i];
                let ydiff = y1[i] - yd0;
                let bspl = h * k1[i] - ydiff;
                cont[0][i] = yd0;
                cont[1][i] = ydiff;
                cont[2][i] = bspl;
                cont[3][i] = -h * k2[i] + ydiff - bspl;
                cont[4][i] = h * (D1 * k1[i] + D3 * k3[i] + D4 * k4[i] + D5 * k5[i] + D6 * k6[i] + D7 * k2[i]);
            }

            // Update state variables
            for i in 0..N {
                k1[i] = k2[i];
                y[i] = y1[i];
            }
            xold = x;
            x = xph;

            match solout.solout(naccpt + 1, xold, x, &y, &cont, h) {
                ControlFlag::Interrupt => {
                    status = Status::Interrupted;
                    break;
                }
                ControlFlag::ModifiedSolution => {
                    f.ode(x, &y, &mut k1);
                    nfcns += 1;
                }
                ControlFlag::Continue => {}
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
            // step rejected
            hnew = h / facc1.min(fac11 / safety_factor);
            reject = true;
            if naccpt > 1 {
                nrejct += 1;
            }
            last = false;
        }
        h = hnew;
        continue;
    }

    DPResult {
        x,
        y,
        h,
        status,
        nfcns,
        nstep,
        naccpt,
        nrejct,
    }
}

/// Evaluate DOPRI5 dense output polynomial for a single abscissa.
pub fn contd5<const N: usize>(cont: &[[Float; N]; 5], xold: Float, h: Float, xi: Float, yi: &mut [Float; N]) {
    let theta = (xi - xold) / h;
    let theta1 = 1.0 - theta;
    for i in 0..N {
        yi[i] = cont[0][i] + theta * (cont[1][i] + theta1 * (cont[2][i] + theta * (cont[3][i] + theta1 * cont[4][i])));
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
