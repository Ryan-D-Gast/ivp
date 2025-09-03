//! Compute an initial step size guess

use crate::{Float, ode::ODE, args::Tolerance};

/// Compute an initial step size guess for an ODE solver.
pub fn hinit<F>(
    f: &F,
    x: Float,
    y: &[Float],
    posneg: Float,
    f0: &[Float],
    f1: &mut [Float],
    y1: &mut [Float],
    iord: usize,
    hmax: Float,
    atol: &Tolerance,
    rtol: &Tolerance,
) -> Float
where
    F: ODE,
{
    let n = y.len();
    let mut dnf: Float = 0.0;
    let mut dny: Float = 0.0;

    for i in 0..n {
        let sk = atol[i] + rtol[i] * y[i].abs();
        dnf += (f0[i] / sk) * (f0[i] / sk);
        dny += (y[i] / sk) * (y[i] / sk);
    }

    let mut h: Float;
    if dnf <= 1e-10 || dny <= 1e-10 {
        h = 1.0e-6;
    } else {
        h = (dny / dnf).sqrt() * 0.01;
    }

    if h > hmax.abs() {
        h = hmax.abs();
    }
    h = h.abs() * posneg.signum();

    // Explicit Euler step: y1 = y + h * f0
    for i in 0..n {
        y1[i] = y[i] + h * f0[i];
    }
    // Evaluate f at x+h
    f.ode(x + h, y1, f1);

    // Estimate second derivative
    let mut der2: Float = 0.0;
    for i in 0..n {
        let sk = atol[i] + rtol[i] * y[i].abs();
        let df = (f1[i] - f0[i]) / sk;
        der2 += df * df;
    }
    der2 = der2.sqrt() / h.abs();

    let der12 = der2.abs().max(dnf.sqrt());
    let h1: Float;
    if der12 <= 1.0e-15_f64 {
        h1 = (1.0e-6_f64).max(h.abs() * 1.0e-3_f64);
    } else {
        h1 = (0.01_f64 / der12).powf(1.0_f64 / (iord as Float));
    }

    let h_final = h.abs().min(100.0_f64 * h.abs()).min(h1).min(hmax.abs());
    h_final.abs() * posneg.signum()
}
