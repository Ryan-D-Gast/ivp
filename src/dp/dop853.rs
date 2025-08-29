//! DOP853 - Dormand–Prince 8(5,3) explicit Runge–Kutta integrator
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
    dp::{DPResult, DPSettings, hinit},
    ode::ODE,
    solout::{ControlFlag, Interpolate, SolOut},
    status::Status,
    tolerance::Tolerance,
};

/// Numerical solution of a system of first order
/// ordinary differential equations in the form
/// `y' = f(x, y)`. This is an explicit Runge-Kutta
/// method of order 8(5,3) due to dormand & prince
/// (with stepsize control and dense output).
pub fn dop853<const N: usize, F, S, R, A>(
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
        fac1 = 1.0 / 3.0;
    }
    if fac2 == 0.0 {
        fac2 = 6.0;
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

    // --- Call DOP853 Core Solver ---
    let rtol = rtol.into();
    let atol = atol.into();
    let result = dp853co::<N, F, S>(
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

/// DOP853 core solver
fn dp853co<const N: usize, F, S>(
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
    let mut k = [[0.0; N]; 10];
    let mut y1 = [0.0; N];
    let mut cont = [[0.0; N]; 8];
    let mut nonstiff = 0;
    let mut facold: Float = 1e-4;
    let mut last = false;
    let mut reject = false;
    let mut hlamb = 0.0;
    let mut iasti = 0;
    let mut xold;
    let mut err;
    let mut err2;
    let mut deno;
    let mut fac;
    let mut hnew;
    let mut fac11;
    let mut sk;
    let mut erri;
    let mut xph;
    let status;
    let expo1 = 1.0 / 8.0 - beta * 0.2;
    let facc1 = 1.0 / fac1;
    let facc2 = 1.0 / fac2;
    let posneg = (xend - x).signum();
    let iord = 8;

    // Initial preparation
    f.ode(x, &y, &mut k[0]);
    nfcns += 1;
    if h == 0.0 {
        let (kl, kr) = k.split_at_mut(1);
        let k0 = &mut kl[0];
        let k1 = &mut kr[0];
        h = hinit(f, x, &y, posneg, k0, k1, &mut y1, iord, hmax, &atol, &rtol);
        nfcns += 1;
    }
    xold = x;
    solout.solout(xold, x, &y, &DenseOutput::new(&cont, &xold, &h));

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

        // --- The twelve stages ---
        // Stage 2
        for i in 0..N {
            y1[i] = y[i] + h * A21 * k[0][i];
        }
        f.ode(x + C2 * h, &y1, &mut k[1]);
        // Stage 3
        for i in 0..N {
            y1[i] = y[i] + h * (A31 * k[0][i] + A32 * k[1][i]);
        }
        f.ode(x + C3 * h, &y1, &mut k[2]);

        // Stage 4
        for i in 0..N {
            y1[i] = y[i] + h * (A41 * k[0][i] + A43 * k[2][i]);
        }
        f.ode(x + C4 * h, &y1, &mut k[3]);

        // Stage 5
        for i in 0..N {
            y1[i] = y[i] + h * (A51 * k[0][i] + A53 * k[2][i] + A54 * k[3][i]);
        }
        f.ode(x + C5 * h, &y1, &mut k[4]);

        // Stage 6
        for i in 0..N {
            y1[i] = y[i] + h * (A61 * k[0][i] + A64 * k[3][i] + A65 * k[4][i]);
        }
        f.ode(x + C6 * h, &y1, &mut k[5]);

        // Stage 7
        for i in 0..N {
            y1[i] = y[i] + h * (A71 * k[0][i] + A74 * k[3][i] + A75 * k[4][i] + A76 * k[5][i]);
        }
        f.ode(x + C7 * h, &y1, &mut k[6]);

        // Stage 8
        for i in 0..N {
            y1[i] = y[i]
                + h * (A81 * k[0][i]
                    + A84 * k[3][i]
                    + A85 * k[4][i]
                    + A86 * k[5][i]
                    + A87 * k[6][i]);
        }
        f.ode(x + C8 * h, &y1, &mut k[7]);

        // Stage 9
        for i in 0..N {
            y1[i] = y[i]
                + h * (A91 * k[0][i]
                    + A94 * k[3][i]
                    + A95 * k[4][i]
                    + A96 * k[5][i]
                    + A97 * k[6][i]
                    + A98 * k[7][i]);
        }
        f.ode(x + C9 * h, &y1, &mut k[8]);

        // Stage 10
        for i in 0..N {
            y1[i] = y[i]
                + h * (A101 * k[0][i]
                    + A104 * k[3][i]
                    + A105 * k[4][i]
                    + A106 * k[5][i]
                    + A107 * k[6][i]
                    + A108 * k[7][i]
                    + A109 * k[8][i]);
        }
        f.ode(x + C10 * h, &y1, &mut k[9]);

        // Stage 11
        for i in 0..N {
            y1[i] = y[i]
                + h * (A111 * k[0][i]
                    + A114 * k[3][i]
                    + A115 * k[4][i]
                    + A116 * k[5][i]
                    + A117 * k[6][i]
                    + A118 * k[7][i]
                    + A119 * k[8][i]
                    + A1110 * k[9][i]);
        }
        f.ode(x + C11 * h, &y1, &mut k[1]);

        // Stage 12
        xph = x + h;
        for i in 0..N {
            y1[i] = y[i]
                + h * (A121 * k[0][i]
                    + A124 * k[3][i]
                    + A125 * k[4][i]
                    + A126 * k[5][i]
                    + A127 * k[6][i]
                    + A128 * k[7][i]
                    + A129 * k[8][i]
                    + A1210 * k[9][i]
                    + A1211 * k[1][i]);
        }
        f.ode(xph, &y1, &mut k[2]);
        nfcns += 11;

        for i in 0..N {
            k[3][i] = B1 * k[0][i]
                + B6 * k[5][i]
                + B7 * k[6][i]
                + B8 * k[7][i]
                + B9 * k[8][i]
                + B10 * k[9][i]
                + B11 * k[1][i]
                + B12 * k[2][i];
            k[4][i] = y[i] + h * k[3][i];
        }

        // Error estimation
        err = 0.0;
        err2 = 0.0;
        for i in 0..N {
            sk = atol[i] + rtol[i] * y[i].abs().max(k[4][i].abs());

            // ERR2 uses K4 - BHH1*K1 - BHH2*K9 - BHH3*K3
            erri = k[3][i] - BH1 * k[0][i] - BH2 * k[8][i] - BH3 * k[2][i];
            err2 += (erri / sk).powi(2);

            // ERRI = er1*K1 + er6*K6 + er7*K7 + er8*K8 + er9*K9 + er10*K10 + er11*K2 + er12*K3
            erri = ER1 * k[0][i]
                + ER6 * k[5][i]
                + ER7 * k[6][i]
                + ER8 * k[7][i]
                + ER9 * k[8][i]
                + ER10 * k[9][i]
                + ER11 * k[1][i]
                + ER12 * k[2][i];
            err += (erri / sk).powi(2);
        }
        deno = err + 0.01 * err2;
        if deno <= 0.0 {
            deno = 1.0;
        }
        let n = N as f64;
        err = h.abs() * err * (1.0 / (n * deno)).sqrt();

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
            let (kl, kr) = k.split_at_mut(4);
            // kl[3] is K4, kr[0] is K5
            f.ode(xph, &kr[0], &mut kl[3]);
            nfcns += 1;

            // Stiffness detection
            if (naccpt % nstiff == 0) || (iasti > 0) {
                let mut stnum: Float = 0.0;
                let mut stden: Float = 0.0;
                for i in 0..N {
                    let d1 = k[3][i] - k[2][i];
                    let d2 = k[4][i] - y1[i];
                    stnum += d1 * d1;
                    stden += d2 * d2;
                }
                if stden > 0.0 {
                    hlamb = h.abs() * (stnum / stden).sqrt();
                }
                if hlamb > 6.1 {
                    nonstiff = 0;
                    iasti += 1;
                    if iasti == 15 {
                        status = Status::ProbablyStiff;
                        break;
                    }
                } else {
                    nonstiff += 1;
                    if nonstiff == 6 {
                        iasti = 0;
                    }
                }
            }

            // Dense output coefficient computation
            for i in 0..N {
                cont[0][i] = y[i];
                let ydiff = k[4][i] - y[i];
                cont[1][i] = ydiff;
                let bspl = h * k[0][i] - ydiff;
                cont[2][i] = bspl;
                cont[3][i] = ydiff - h * k[3][i] - bspl;
                cont[4][i] = D41 * k[0][i]
                    + D46 * k[5][i]
                    + D47 * k[6][i]
                    + D48 * k[7][i]
                    + D49 * k[8][i]
                    + D410 * k[9][i]
                    + D411 * k[1][i]
                    + D412 * k[2][i];

                cont[5][i] = D51 * k[0][i]
                    + D56 * k[5][i]
                    + D57 * k[6][i]
                    + D58 * k[7][i]
                    + D59 * k[8][i]
                    + D510 * k[9][i]
                    + D511 * k[1][i]
                    + D512 * k[2][i];

                cont[6][i] = D61 * k[0][i]
                    + D66 * k[5][i]
                    + D67 * k[6][i]
                    + D68 * k[7][i]
                    + D69 * k[8][i]
                    + D610 * k[9][i]
                    + D611 * k[1][i]
                    + D612 * k[2][i];

                cont[7][i] = D71 * k[0][i]
                    + D76 * k[5][i]
                    + D77 * k[6][i]
                    + D78 * k[7][i]
                    + D79 * k[8][i]
                    + D710 * k[9][i]
                    + D711 * k[1][i]
                    + D712 * k[2][i];
            }

            // Next three function evaluations
            for i in 0..N {
                y1[i] = y[i]
                    + h * (A141 * k[0][i]
                        + A147 * k[6][i]
                        + A148 * k[7][i]
                        + A149 * k[8][i]
                        + A1410 * k[9][i]
                        + A1411 * k[1][i]
                        + A1412 * k[2][i]
                        + A1413 * k[3][i]);
            }
            f.ode(x + C14 * h, &y1, &mut k[9]);

            for i in 0..N {
                y1[i] = y[i]
                    + h * (A151 * k[0][i]
                        + A156 * k[5][i]
                        + A157 * k[6][i]
                        + A158 * k[7][i]
                        + A1511 * k[1][i]
                        + A1512 * k[2][i]
                        + A1513 * k[3][i]
                        + A1514 * k[9][i]);
            }
            f.ode(x + C15 * h, &y1, &mut k[1]);

            for i in 0..N {
                y1[i] = y[i]
                    + h * (A161 * k[0][i]
                        + A166 * k[5][i]
                        + A167 * k[6][i]
                        + A168 * k[7][i]
                        + A169 * k[8][i]
                        + A1613 * k[3][i]
                        + A1614 * k[9][i]
                        + A1615 * k[1][i]);
            }
            f.ode(x + C16 * h, &y1, &mut k[2]);
            nfcns += 3;

            // Final dense output coefficients
            for i in 0..N {
                cont[4][i] = h
                    * (cont[4][i]
                        + D413 * k[3][i]
                        + D414 * k[9][i]
                        + D415 * k[1][i]
                        + D416 * k[2][i]);

                cont[5][i] = h
                    * (cont[5][i]
                        + D513 * k[3][i]
                        + D514 * k[9][i]
                        + D515 * k[1][i]
                        + D516 * k[2][i]);

                cont[6][i] = h
                    * (cont[6][i]
                        + D613 * k[3][i]
                        + D614 * k[9][i]
                        + D615 * k[1][i]
                        + D616 * k[2][i]);

                cont[7][i] = h
                    * (cont[7][i]
                        + D713 * k[3][i]
                        + D714 * k[9][i]
                        + D715 * k[1][i]
                        + D716 * k[2][i]);
            }

            // Update state variables
            for i in 0..N {
                k[0][i] = k[3][i];
                y[i] = k[4][i];
            }
            xold = x;
            x = xph;

            match solout.solout(xold, x, &y, &DenseOutput::new(&cont, &xold, &h)) {
                ControlFlag::Interrupt => {
                    status = Status::Interrupted;
                    break;
                }
                ControlFlag::ModifiedSolution => {
                    f.ode(x, &y, &mut k[0]);
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

/// Dense output interpolator for DOP853
struct DenseOutput<'a, const N: usize> {
    cont: &'a [[Float; N]; 8],
    xold: &'a Float,
    h: &'a Float,
}

impl<'a, const N: usize> DenseOutput<'a, N> {
    fn new(cont: &'a [[Float; N]; 8], xold: &'a Float, h: &'a Float) -> Self {
        Self { cont, xold, h }
    }
}

impl<'a, const N: usize> Interpolate<N> for DenseOutput<'a, N> {
    fn interpolate(&self, xi: Float) -> [Float; N] {
        let mut yi = [0.0; N];
        let s = (xi - *self.xold) / *self.h;
        let s1 = 1.0 - s;
        for i in 0..N {
            let conpar = self.cont[4][i]
                + s * (self.cont[5][i] + s1 * (self.cont[6][i] + s * self.cont[7][i]));
            let contd8 = self.cont[0][i]
                + s * (self.cont[1][i]
                    + s1 * (self.cont[2][i] + s * (self.cont[3][i] + s1 * conpar)));
            yi[i] = contd8;
        }
        yi
    }
}

// DOP853 Butcher tableau coefficients
const C2: Float = 0.526001519587677318785587544488e-01;
const C3: Float = 0.789002279381515978178381316732e-01;
const C4: Float = 0.118350341907227396726757197510e+00;
const C5: Float = 0.281649658092772603273242802490e+00;
const C6: Float = 0.333333333333333333333333333333e+00;
const C7: Float = 0.25e+00;
const C8: Float = 0.307692307692307692307692307692e+00;
const C9: Float = 0.651282051282051282051282051282e+00;
const C10: Float = 0.6e+00;
const C11: Float = 0.857142857142857142857142857142e+00;
const C14: Float = 0.1e+00;
const C15: Float = 0.2e+00;
const C16: Float = 7.777_777_777_777_778e-1;

const A21: Float = 5.26001519587677318785587544488e-2;

const A31: Float = 1.97250569845378994544595329183e-2;
const A32: Float = 5.91751709536136983633785987549e-2;

const A41: Float = 2.95875854768068491816892993775e-2;
const A43: Float = 8.87627564304205475450678981324e-2;

const A51: Float = 2.41365134159266685502369798665e-1;
const A53: Float = -8.84549479328286085344864962717e-1;
const A54: Float = 9.24834003261792003115737966543e-1;

const A61: Float = 3.7037037037037037037037037037e-2;
const A64: Float = 1.70828608729473871279604482173e-1;
const A65: Float = 1.25467687566822425016691814123e-1;

const A71: Float = 3.7109375e-2;
const A74: Float = 1.70252211019544039314978060272e-1;
const A75: Float = 6.02165389804559606850219397283e-2;
const A76: Float = -1.7578125e-2;

const A81: Float = 3.70920001185047927108779319836e-2;
const A84: Float = 1.70383925712239993810214054705e-1;
const A85: Float = 1.07262030446373284651809199168e-1;
const A86: Float = -1.53194377486244017527936158236e-2;
const A87: Float = 8.27378916381402288758473766002e-3;

const A91: Float = 6.24110958716075717114429577812e-1;
const A94: Float = -3.36089262944694129406857109825e0;
const A95: Float = -8.68219346841726006818189891453e-1;
const A96: Float = 2.75920996994467083049415600797e1;
const A97: Float = 2.01540675504778934086186788979e1;
const A98: Float = -4.34898841810699588477366255144e1;

const A101: Float = 4.77662536438264365890433908527e-1;
const A104: Float = -2.48811461997166764192642586468e0;
const A105: Float = -5.90290826836842996371446475743e-1;
const A106: Float = 2.12300514481811942347288949897e1;
const A107: Float = 1.52792336328824235832596922938e1;
const A108: Float = -3.32882109689848629194453265587e1;
const A109: Float = -2.03312017085086261358222928593e-2;

const A111: Float = -9.3714243008598732571704021658e-1;
const A114: Float = 5.18637242884406370830023853209e0;
const A115: Float = 1.09143734899672957818500254654e0;
const A116: Float = -8.14978701074692612513997267357e0;
const A117: Float = -1.85200656599969598641566180701e1;
const A118: Float = 2.27394870993505042818970056734e1;
const A119: Float = 2.49360555267965238987089396762e0;
const A1110: Float = -3.0467644718982195003823669022e0;

const A121: Float = 2.27331014751653820792359768449e0;
const A124: Float = -1.05344954667372501984066689879e1;
const A125: Float = -2.00087205822486249909675718444e0;
const A126: Float = -1.79589318631187989172765950534e1;
const A127: Float = 2.79488845294199600508499808837e1;
const A128: Float = -2.85899827713502369474065508674e0;
const A129: Float = -8.87285693353062954433549289258e0;
const A1210: Float = 1.23605671757943030647266201528e1;
const A1211: Float = 6.43392746015763530355970484046e-1;

const B1: Float = 5.42937341165687622380535766363e-2;
const B6: Float = 4.45031289275240888144113950566e0;
const B7: Float = 1.89151789931450038304281599044e0;
const B8: Float = -5.8012039600105847814672114227e0;
const B9: Float = 3.1116436695781989440891606237e-1;
const B10: Float = -1.52160949662516078556178806805e-1;
const B11: Float = 2.01365400804030348374776537501e-1;
const B12: Float = 4.47106157277725905176885569043e-2;

const BH1: Float = 0.244094488188976377952755905512e+00;
const BH2: Float = 0.733846688281611857341361741547e+00;
const BH3: Float = 0.220588235294117647058823529412e-01;

const ER1: Float = 0.1312004499419488073250102996e-01;
const ER6: Float = -0.1225156446376204440720569753e+01;
const ER7: Float = -0.4957589496572501915214079952e+00;
const ER8: Float = 0.1664377182454986536961530415e+01;
const ER9: Float = -0.3503288487499736816886487290e+00;
const ER10: Float = 0.3341791187130174790297318841e+00;
const ER11: Float = 0.8192320648511571246570742613e-01;
const ER12: Float = -0.2235530786388629525884427845e-01;

const A141: Float = 5.61675022830479523392909219681e-2;
const A147: Float = 2.53500210216624811088794765333e-1;
const A148: Float = -2.46239037470802489917441475441e-1;
const A149: Float = -1.24191423263816360469010140626e-1;
const A1410: Float = 1.5329179827876569731206322685e-1;
const A1411: Float = 8.20105229563468988491666602057e-3;
const A1412: Float = 7.56789766054569976138603589584e-3;
const A1413: Float = -8.298e-3;

const A151: Float = 3.18346481635021405060768473261e-2;
const A156: Float = 2.83009096723667755288322961402e-2;
const A157: Float = 5.35419883074385676223797384372e-2;
const A158: Float = -5.49237485713909884646569340306e-2;
const A1511: Float = -1.08347328697249322858509316994e-4;
const A1512: Float = 3.82571090835658412954920192323e-4;
const A1513: Float = -3.40465008687404560802977114492e-4;
const A1514: Float = 1.41312443674632500278074618366e-1;

const A161: Float = -4.28896301583791923408573538692e-1;
const A166: Float = -4.69762141536116384314449447206e0;
const A167: Float = 7.68342119606259904184240953878e0;
const A168: Float = 4.06898981839711007970213554331e0;
const A169: Float = 3.56727187455281109270669543021e-1;
const A1613: Float = -1.39902416515901462129418009734e-3;
const A1614: Float = 2.9475147891527723389556272149e0;
const A1615: Float = -9.15095847217987001081870187138e0;

const D41: Float = -0.84289382761090128651353491142e+01;
const D46: Float = 0.56671495351937776962531783590e+00;
const D47: Float = -0.30689499459498916912797304727e+01;
const D48: Float = 0.23846676565120698287728149680e+01;
const D49: Float = 0.21170345824450282767155149946e+01;
const D410: Float = -0.87139158377797299206789907490e+00;
const D411: Float = 0.22404374302607882758541771650e+01;
const D412: Float = 0.63157877876946881815570249290e+00;
const D413: Float = -0.88990336451333310820698117400e-01;
const D414: Float = 0.18148505520854727256656404962e+02;
const D415: Float = -0.91946323924783554000451984436e+01;
const D416: Float = -0.44360363875948939664310572000e+01;

const D51: Float = 0.10427508642579134603413151009e+02;
const D56: Float = 0.24228349177525818288430175319e+03;
const D57: Float = 0.16520045171727028198505394887e+03;
const D58: Float = -0.37454675472269020279518312152e+03;
const D59: Float = -0.22113666853125306036270938578e+02;
const D510: Float = 0.77334326684722638389603898808e+01;
const D511: Float = -0.30674084731089398182061213626e+02;
const D512: Float = -0.93321305264302278729567221706e+01;
const D513: Float = 0.15697238121770843886131091075e+02;
const D514: Float = -0.31139403219565177677282850411e+02;
const D515: Float = -0.93529243588444783865713862664e+01;
const D516: Float = 0.35816841486394083752465898540e+02;

const D61: Float = 0.19985053242002433820987653617e+02;
const D66: Float = -0.38703730874935176555105901742e+03;
const D67: Float = -0.18917813819516756882830838328e+03;
const D68: Float = 0.52780815920542364900561016686e+03;
const D69: Float = -0.11573902539959630126141871134e+02;
const D610: Float = 0.68812326946963000169666922661e+01;
const D611: Float = -0.10006050966910838403183860980e+01;
const D612: Float = 0.77771377980534432092869265740e+00;
const D613: Float = -0.27782057523535084065932004339e+01;
const D614: Float = -0.60196695231264120758267380846e+02;
const D615: Float = 0.84320405506677161018159903784e+02;
const D616: Float = 0.11992291136182789328035130030e+02;

const D71: Float = -0.25693933462703749003312586129e+02;
const D76: Float = -0.15418974869023643374053993627e+03;
const D77: Float = -0.23152937917604549567536039109e+03;
const D78: Float = 0.35763911791061412378285349910e+03;
const D79: Float = 0.93405324183624310003907691704e+02;
const D710: Float = -0.37458323136451633156875139351e+02;
const D711: Float = 0.10409964950896230045147246184e+03;
const D712: Float = 0.29840293426660503123344363579e+02;
const D713: Float = -0.43533456590011143754432175058e+02;
const D714: Float = 0.96324553959188282948394950600e+02;
const D715: Float = -0.39177261675615439165231486172e+02;
const D716: Float = -0.14972683625798562581422125276e+03;
