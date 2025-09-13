//! Radau5 — 3-stage, order-5 Radau IIA implicit Runge–Kutta solver.
//!
//! Solves stiff ODEs/DAEs `M·y' = f(t,y)` with adaptive step-size,
//! simplified Newton iterations (numerical Jacobian by default), and dense output.
//! Reference: Hairer & Wanner, Solving ODEs II (Radau IIA).

use crate::{
    Float,
    error::Error,
    interpolate::Interpolate,
    matrix::{Matrix, lin_solve, lin_solve_complex, lu_decomp, lu_decomp_complex},
    methods::{
        result::IntegrationResult,
        settings::{Settings, Tolerance},
    },
    ode::ODE,
    solout::{ControlFlag, SolOut},
    status::Status,
};

/// Radau IIA(5) implicit Runge–Kutta with adaptive steps and dense output.
pub fn radau5<F, S>(
    f: &F,
    mut x: Float,
    xend: Float,
    y0: &[Float],
    mut rtol: Tolerance,
    mut atol: Tolerance,
    mut solout: Option<&mut S>,
    settings: Settings,
) -> Result<IntegrationResult, Vec<Error>>
where
    F: ODE,
    S: SolOut,
{
    // --- Input Validation ---
    let mut errors: Vec<Error> = Vec::new();

    // nmax
    let nmax = settings.nmax.unwrap_or(100_000);
    if nmax == 0 {
        errors.push(Error::NMaxMustBePositive(0));
    }
    // uround
    let uround = match settings.uround {
        Some(u) if (1e-35..1.0).contains(&u) => u,
        Some(u) => {
            errors.push(Error::URoundOutOfRange(u));
            u
        }
        None => 2.3e-16,
    };
    // safety factor
    let safety_factor = match settings.safety_factor {
        Some(s) if s > 1e-4 && s < 1.0 => s,
        Some(s) => {
            errors.push(Error::SafetyFactorOutOfRange(s));
            s
        }
        None => 0.9,
    };
    // Step-size scaling bounds: clamp factor quot in [facc2, facc1]
    let scale_min = settings.scale_min.unwrap_or(0.2);
    let scale_max = settings.scale_max.unwrap_or(5.0);
    let facc1 = 1.0 / scale_min;
    let facc2 = 1.0 / scale_max;
    if scale_min <= 0.0 || !(scale_min < scale_max) {
        errors.push(Error::InvalidScaleFactors(scale_min, scale_max));
    }
    let n = y0.len();
    let mut y = y0.to_vec();

    // hmax and hmin
    let hmax = settings.hmax.unwrap_or_else(|| (xend - x).abs());
    let hmin = settings.hmin.unwrap_or(0.0);

    // Max newton iterations
    let max_newton = settings.newton_maxiter.unwrap_or(7);
    if max_newton <= 0 {
        errors.push(Error::NewtonMaxIterMustBePositive(0));
    }

    // Newton tolerance
    let newton_tol = settings.newton_tol.unwrap_or(0.003_162_277_660_168_379_4);

    if !errors.is_empty() {
        return Err(errors);
    }

    // --- Initialization ---

    // Adjust tolerances
    let expm = 2.0 / 3.0;
    for i in 0..n {
        let quot = atol[i] / rtol[i];
        rtol[i] = 0.1 * rtol[i].powf(expm);
        atol[i] = rtol[i] * quot;
    }

    let posneg = (xend - x).signum();
    let mut f0 = vec![0.0; n];
    f.ode(x, &y, &mut f0);
    let mut nfev: usize = 1;

    // Initial step size: use provided h0 or default to 1e-6 (signed)
    let mut h = if let Some(h0) = settings.h0 {
        h0
    } else {
        1.0e-6 * posneg
    };
    if h == 0.0 || h.signum() != posneg && posneg != 0.0 {
        return Err(vec![Error::InvalidStepSize(h)]);
    }
    h = h.clamp(-hmax, hmax);

    // Dense output: [y_{n+1}, c1, c2, c3]
    let mut cont = vec![0.0; n * 4];

    // Initial callback (xold=x)
    if let Some(s) = solout.as_mut() {
        let interp = DenseRadau {
            cont: &cont,
            xold: x,
            h,
        };
        s.solout(x, x, &y, &interp);
    }

    // Workspace
    let mut z1 = vec![0.0; n];
    let mut z2 = vec![0.0; n];
    let mut z3 = vec![0.0; n];
    let mut f1 = vec![0.0; n];
    let mut f2 = vec![0.0; n];
    let mut f3 = vec![0.0; n];
    let mut scal = vec![0.0; n];
    let mut jac = Matrix::zeros(n, n);
    let mut mass = Matrix::identity(n);
    let mut e1 = Matrix::zeros(n, n);
    let mut e2r = Matrix::zeros(n, n);
    let mut e2i = Matrix::zeros(n, n);
    let mut ip1 = vec![0; n];
    let mut ip2 = vec![0; n];

    let mut nstep: usize = 0;
    let mut naccpt: usize = 0;
    let mut nrejct: usize = 0;
    let mut nsol: usize = 0;
    let mut ndec: usize = 0;
    let mut njac: usize = 0;
    let mut status = Status::Success;
    let mut last = false;
    let mut singular_count = 0;
    let mut faccon: Float = 1.0;
    let mut theta: Float;
    let mut dynold: Float = 0.0;
    let mut thqold: Float = 0.0;
    let mut err: Float;
    let mut hnew: Float;
    let mut reject = false;
    let mut first = true;
    let mut call_jac = true;
    let mut call_decomp = true;
    let mut qt;
    let mut hold = h;
    let mut fac: Float;
    let mut xold;
    let mut quot: Float;
    let mut h_acc: Float = 0.0;
    let mut hhfac: Float = h;
    let mut err_acc: Float = 0.0;
    let mut xph;
    let mut dyno: Float;
    let quot1: Float = 1.0;
    let quot2: Float = 1.2;
    let thet: Float = 0.001;
    let cfac: Float = 13.5;
    let predictive = true;

    // Temp index
    let index2 = false;
    let index3 = false;
    let nind1 = 0;
    let nind2 = 0;
    let nind3 = 0;

    // Initial mass matrix
    f.mass(&mut mass);

    // Error scale
    for i in 0..n {
        scal[i] = atol[i] + rtol[i] * y[i].abs();
    }

    // --- Main loop ---
    'main: loop {
        if posneg * (xend - x) <= 0.0 {
            break;
        }

        if call_jac {
            // Jacobian and mass at (x, y)
            f.jac(x, &y, &mut jac);
            njac += 1;
        }

        if call_decomp {
            // Build E1 and E2 matrices
            let fac1 = U1 / h;
            let alphn = ALPH / h;
            let betan = BETA / h;
            for r in 0..n {
                for c in 0..n {
                    // E1 = (U1/h)·M − J
                    e1[(r, c)] = mass[(r, c)] * fac1 - jac[(r, c)];
                    e2r[(r, c)] = mass[(r, c)] * alphn - jac[(r, c)];
                    e2i[(r, c)] = mass[(r, c)] * betan;
                }
            }

            // LU decomp of real matrix E1
            if lu_decomp(&mut e1, &mut ip1).is_err() {
                singular_count += 1;
                if singular_count > 5 {
                    status = Status::SingularMatrix;
                    break 'main;
                }
                h *= 0.5;
                hhfac = 0.5;
                reject = true;
                last = false;
                continue 'main;
            }

            // LU decomp of complex matrix E2
            if lu_decomp_complex(&mut e2r, &mut e2i, &mut ip2).is_err() {
                singular_count += 1;
                if singular_count > 5 {
                    status = Status::SingularMatrix;
                    break 'main;
                }
                h *= 0.5;
                hhfac = 0.5;
                reject = true;
                last = false;
                continue 'main;
            }
            // Count both decompositions
            ndec += 2;
        }

        // --- Integration step ---
        nstep += 1;

        // Max step guard
        if nstep > nmax {
            status = Status::NeedLargerNmax;
            break;
        }

        // Step size guard
        if 0.1 * h.abs() <= x.abs() * uround {
            status = Status::StepSizeTooSmall;
            break;
        }

        // Account for index 2 and 3 algebraic variables
        if index2 {
            for i in nind1..(nind1 + nind2) {
                scal[i] /= hhfac;
            }
        }
        if index3 {
            for i in (nind1 + nind2)..(nind1 + nind2 + nind3) {
                scal[i] /= hhfac * hhfac;
            }
        }
        xph = x + h;

        // Initialize stage increments and transforms
        if first {
            for i in 0..n {
                z1[i] = 0.0;
                z2[i] = 0.0;
                z3[i] = 0.0;
                f1[i] = 0.0;
                f2[i] = 0.0;
                f3[i] = 0.0;
            }
        } else {
            let c3q = h / hold;
            let c1q = C1 * c3q;
            let c2q = C2 * c3q;

            for i in 0..n {
                let ak1 = cont[1 * n + i];
                let ak2 = cont[2 * n + i];
                let ak3 = cont[3 * n + i];

                z1[i] = (ak1 + (ak2 + ak3 * (c1q - C1M1)) * (c1q - C2M1)) * c1q;
                z2[i] = (ak1 + (ak2 + ak3 * (c2q - C1M1)) * (c2q - C2M1)) * c2q;
                z3[i] = (ak1 + (ak2 + ak3 * (c3q - C1M1)) * (c3q - C2M1)) * c3q;

                f1[i] = z1[i] * TI00 + z2[i] * TI01 + z3[i] * TI02;
                f2[i] = z1[i] * TI10 + z2[i] * TI11 + z3[i] * TI12;
                f3[i] = z1[i] * TI20 + z2[i] * TI21 + z3[i] * TI22;
            }
        }

        // --- Loop for simplified newton iteration ---
        faccon = faccon.max(uround).powf(0.8);
        theta = thet.abs();
        let mut newt_iter = 0;
        'newton: loop {
            if newt_iter >= max_newton {
                singular_count += 1;
                if singular_count > 5 {
                    status = Status::SingularMatrix;
                    break 'main;
                }
                h *= 0.5;
                hhfac = 0.5;
                reject = true;
                last = false;
                continue 'main;
            }

            // --- Compute the stages ---
            for i in 0..n {
                cont[i] = y[i] + z1[i];
            }
            f.ode(x + C1 * h, &cont, &mut z1);
            for i in 0..n {
                cont[i] = y[i] + z2[i];
            }
            f.ode(x + C2 * h, &cont, &mut z2);
            for i in 0..n {
                cont[i] = y[i] + z3[i];
            }
            f.ode(xph, &cont, &mut z3);
            nfev += 3;

            // --- Solve the linear systems ---
            for i in 0..n {
                z1[i] = TI00 * z1[i] + TI01 * z2[i] + TI02 * z3[i];
                z2[i] = TI10 * z1[i] + TI11 * z2[i] + TI12 * z3[i];
                z3[i] = TI20 * z1[i] + TI21 * z2[i] + TI22 * z3[i];
            }

            // Might be a duplicate.
            let fac1 = U1 / h;
            let alphn = ALPH / h;
            let betan = BETA / h;

            // Add mass contributions from current F
            for i in 0..n {
                let mut sum1 = 0.0;
                let mut sum2 = 0.0;
                let mut sum3 = 0.0;
                for j in 0..n {
                    let mij = mass[(i, j)];
                    sum1 -= mij * f1[j];
                    sum2 -= mij * f2[j];
                    sum3 -= mij * f3[j];
                }
                z1[i] = z1[i] + sum1 * fac1;
                z2[i] = z2[i] + sum2 * alphn - sum3 * betan;
                z3[i] = z3[i] + sum3 * alphn + sum2 * betan;
            }

            // Solve E1 * Z1 = RHS1 (real system)
            lin_solve(&e1, &mut z1, &ip1);

            // Solve E2 * [Z2; Z3] = [RHS2; RHS3] (complex system)
            lin_solve_complex(&e2r, &e2i, &mut z2, &mut z3, &ip2);

            nsol += 2;
            newt_iter += 1;

            // Compute dynamic norm
            dyno = 0.0;
            for i in 0..n {
                let denom = scal[i];
                let v1 = z1[i] / denom;
                let v2 = z2[i] / denom;
                let v3 = z3[i] / denom;
                dyno += v1 * v1 + v2 * v2 + v3 * v3;
            }
            dyno = (dyno / (3.0 * n as Float)).sqrt();

            // Bad convergence or number of iterations is too large
            if newt_iter > 1 && newt_iter < max_newton {
                let thq = dyno / dynold;
                if newt_iter == 2 {
                    theta = thq;
                } else {
                    theta = (thq * thqold).sqrt();
                }
                thqold = thq;
                if theta < 0.99 {
                    faccon = theta / (1.0 - theta);
                    let remaining_iters = (max_newton - 1 - newt_iter) as Float;
                    let dyth = faccon * dyno * theta.powf(remaining_iters) / newton_tol;
                    if dyth >= 1.0 {
                        let qnewt = 1e-4f64.max(20.0f64.min(dyth as f64)) as Float;
                        let exponent = -1.0 / (4.0 + remaining_iters);
                        hhfac = 0.8 * qnewt.powf(exponent);
                        h *= hhfac;
                        status = Status::PoorConvergence; // Using appropriate status
                        nrejct += 1;
                        last = false;
                        break 'newton;
                    }
                } else {
                    // Unexpected step rejection - continue with reduced step
                    singular_count += 1;
                    if singular_count > 5 {
                        status = Status::SingularMatrix;
                        break 'main;
                    }
                    h *= 0.5;
                    hhfac = 0.5;
                    reject = true;
                    last = false;
                    continue 'main;
                }
            }
            dynold = dyno.max(uround);

            // Compute new F and Z
            for i in 0..n {
                f1[i] += z1[i];
                f2[i] += z2[i];
                f3[i] += z3[i];
            }

            for i in 0..n {
                z1[i] = f1[i] * T00 + f2[i] * T01 + f3[i] * T02;
                z2[i] = f1[i] * T10 + f2[i] * T11 + f3[i] * T12;
                z3[i] = f1[i] * T20 + f2[i];
            }

            // Check Newton tolerance
            if faccon * dyno > newton_tol {
                continue 'newton;
            } else {
                break 'newton;
            }
        }

        // --- Error estimation ---
        let hee1 = DD1 / h;
        let hee2 = DD2 / h;
        let hee3 = DD3 / h;
        for i in 0..n {
            f1[i] = hee1 * z1[i] + hee2 * z2[i] + hee3 * z3[i];
            f2[i] = 0.0;
        }
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += mass[(i, j)] * f1[j];
            }
            f2[i] = sum;
            cont[i] = sum + f0[i];
        }
        lin_solve(&e1, &mut cont, &ip1);
        nsol += 1;

        err = 0.0;
        for i in 0..n {
            let r = cont[i] / scal[i];
            err += r * r;
        }
        err = (err / n as Float).sqrt().max(1e-10);

        // Optional refinement on first/rejected step
        if err >= 1.0 && (first || reject) {
            for i in 0..n {
                cont[i] = y[i] + cont[i];
            }
            f.ode(x, &cont, &mut f1);
            nfev += 1;

            // contv = f1 + f2; solve again
            for i in 0..n {
                cont[i] = f1[i] + f2[i];
            }
            lin_solve(&e1, &mut cont, &ip1);
            nsol += 1;

            // Recompute error
            err = 0.0;
            for i in 0..n {
                let r = cont[i] / scal[i];
                err += r * r;
            }
            err = (err / n as Float).sqrt().max(1e-10);
        }

        // --- Computation of hnew ---
        fac = safety_factor.min(cfac / (newt_iter as Float + 2.0 * max_newton as Float));
        quot = facc2.max(facc1.min(err.powf(0.25) / fac));
        hnew = h / quot;

        if err <= 1.0 {
            // --- Step accepted ---
            naccpt += 1;
            first = false;

            // Predictive Gustafsson controller (use previous accepted step if available)
            if predictive {
                if naccpt > 1 {
                    let mut facgus = (h_acc / h) * (err * err / err_acc).powf(0.25) / safety_factor;
                    facgus = facc2.max(facc1.min(facgus));
                    quot = quot.max(facgus);
                    hnew = h / quot;
                }
                h_acc = h;
                err_acc = err.max(1e-2);
            }

            // Update solution
            xold = x;
            hold = h;
            x = xph;

            for i in 0..n {
                y[i] += z3[i];
                let ak = (z1[i] - z2[i]) / C1MC2;
                let acont3 = (ak - (z1[i] / C1)) / C2;
                cont[0 * n + i] = y[i];
                cont[1 * n + i] = (z2[i] - z3[i]) / C2M1;
                cont[2 * n + i] = (ak - cont[1 * n + i]) / C1M1;
                cont[3 * n + i] = cont[2 * n + i] - acont3;
            }

            // Compute error scale
            for i in 0..n {
                scal[i] = atol[i] + rtol[i] * y[i].abs();
            }

            // Callback
            if let Some(ref mut s) = solout {
                match s.solout(
                    xold,
                    x,
                    &y,
                    &DenseRadau {
                        cont: &cont,
                        xold,
                        h,
                    },
                ) {
                    ControlFlag::Continue => {}
                    ControlFlag::Interrupt => {
                        status = Status::Interrupted;
                        break;
                    }
                    ControlFlag::ModifiedSolution(nx, ny) => {
                        x = nx;
                        y.copy_from_slice(&ny);
                        f.ode(x, &y, &mut f0);
                        nfev += 1;
                    }
                }
            }
            call_jac = false;

            if last {
                h = hnew;
                break 'main;
            }

            // New derivative at x+h
            f.ode(x, &y, &mut f0);
            nfev += 1;

            // Constrain new step size
            hnew = hnew.abs().clamp(hmin, hmax) * posneg;

            // Prevent oscillations due to previous step rejections
            if reject {
                hnew = posneg * hnew.abs().min(h.abs());
                reject = false;
            }

            // Sophisticated step size control
            if (x + hnew/quot1 - xend) * posneg >= 0.0 {
                h = xend - x;
                last = true;
            } else {
                qt = hnew / h;
                hhfac = h;
                if theta < thet && qt > quot1 && qt < quot2 {
                    call_decomp = false;
                    call_jac = false;
                    continue 'main;
                }
                h = hnew;
            }
            hhfac = h;
        } else {
            // --- Step rejected ---
            nrejct += 1;
            reject = true;
            last = false;

            // If first step, reduce more aggressively
            if first {
                h *= 0.1;
                hhfac = 0.1;
            } else {
                hhfac = hnew / h;
                h = hnew;
            }
        }
    }

    Ok(IntegrationResult {
        x,
        y,
        h,
        status,
        nfev,
        njac,
        nsol,
        ndec,
        nstep,
        naccpt,
        nrejct,
    })
}

pub fn contr5(xi: Float, yi: &mut [Float], cont: &[Float], xold: Float, h: Float) {
    let n = cont.len() / 4;
    // s = (xi - (xold + h)) / h
    let s = (xi - (xold + h)) / h;
    let c0 = &cont[0 * n..1 * n];
    let c1 = &cont[1 * n..2 * n];
    let c2 = &cont[2 * n..3 * n];
    let c3 = &cont[3 * n..4 * n];
    for i in 0..n {
        yi[i] = c0[i] + s * (c1[i] + (s - C2M1) * (c2[i] + (s - C1M1) * c3[i]));
    }
}

/// Dense output: cubic at the right endpoint using `cont` coefficients.
struct DenseRadau<'a> {
    cont: &'a [Float], // [c0(n)=y_{n+1}, c1(n), c2(n), c3(n)]
    xold: Float,
    h: Float,
}

impl<'a> Interpolate for DenseRadau<'a> {
    fn interpolate(&self, xi: Float, yi: &mut [Float]) {
        contr5(xi, yi, self.cont, self.xold, self.h);
    }

    fn get_cont(&self) -> (Vec<Float>, Float, Float) {
        (self.cont.to_vec(), self.xold, self.h)
    }
}

// Nodes (abscissae) in [0,1]
const C1: Float = 0.155_051_025_721_682_2;
const C2: Float = 0.644_948_974_278_317_8;
const C1M1: Float = C1 - 1.0;
const C2M1: Float = C2 - 1.0;
const C1MC2: Float = C1 - C2;

// Error estimation and splitting constants
const DD1: Float = -10.048_809_399_827_416;
const DD2: Float = 1.382_142_733_160_749;
const DD3: Float = -0.333_333_333_333_333_3;
const U1: Float = 3.637_834_252_744_496; // real system coefficient
const ALPH: Float = 2.681_082_873_627_752_3;
const BETA: Float = 3.050_430_199_247_410_5;

// Transformation matrix T (3x3) constants
const T00: Float = 9.123_239_487_089_295E-2;
const T01: Float = -1.412_552_950_209_542E-1;
const T02: Float = -3.002_919_410_514_742_4E-2;
const T10: Float = 2.417_179_327_071_07E-1;
const T11: Float = 2.041_293_522_937_999_4E-1;
const T12: Float = 3.829_421_127_572_619E-1;
const T20: Float = 9.660_481_826_150_93E-1;

// Inverse transformation matrix T^{-1} constants
const TI00: Float = 4.325_579_890_063_155;
const TI01: Float = 3.391_992_518_158_098_4E-1;
const TI02: Float = 5.417_705_399_358_749E-1;
const TI10: Float = -4.178_718_591_551_905;
const TI11: Float = -3.276_828_207_610_623_7E-1;
const TI12: Float = 4.766_235_545_005_504_4E-1;
const TI20: Float = -5.028_726_349_457_868E-1;
const TI21: Float = 2.571_926_949_855_605;
const TI22: Float = -5.960_392_048_282_249E-1;
