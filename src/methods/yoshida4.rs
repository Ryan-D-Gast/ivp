//! Yoshida fourth-order symplectic composition.

use crate::{Float, ivp::SeparableHamiltonianSystem};

use super::SymplecticWork;

pub(crate) fn step<F>(
    f: &F,
    t: Float,
    h: Float,
    q: &mut [Float],
    p: &mut [Float],
    work: &mut SymplecticWork,
    nfev: &mut usize,
) where
    F: SeparableHamiltonianSystem,
{
    let two_cubert = (2.0f64).powf(1.0 / 3.0) as Float;
    let w1 = 1.0 / (2.0 - two_cubert);
    let w0 = -two_cubert / (2.0 - two_cubert);

    f.momentum_derivative(t, q, &mut work.dpdt);
    *nfev += 1;
    for (pi, dpdt_i) in p.iter_mut().zip(work.dpdt.iter()) {
        *pi += (0.5 * w1) * h * *dpdt_i;
    }

    f.position_derivative(t, p, &mut work.dqdt);
    *nfev += 1;
    for (qi, dqdt_i) in q.iter_mut().zip(work.dqdt.iter()) {
        *qi += w1 * h * *dqdt_i;
    }

    f.momentum_derivative(t, q, &mut work.dpdt);
    *nfev += 1;
    for (pi, dpdt_i) in p.iter_mut().zip(work.dpdt.iter()) {
        *pi += (0.5 * (w1 + w0)) * h * *dpdt_i;
    }

    f.position_derivative(t, p, &mut work.dqdt);
    *nfev += 1;
    for (qi, dqdt_i) in q.iter_mut().zip(work.dqdt.iter()) {
        *qi += w0 * h * *dqdt_i;
    }

    f.momentum_derivative(t, q, &mut work.dpdt);
    *nfev += 1;
    for (pi, dpdt_i) in p.iter_mut().zip(work.dpdt.iter()) {
        *pi += (0.5 * (w0 + w1)) * h * *dpdt_i;
    }

    f.position_derivative(t, p, &mut work.dqdt);
    *nfev += 1;
    for (qi, dqdt_i) in q.iter_mut().zip(work.dqdt.iter()) {
        *qi += w1 * h * *dqdt_i;
    }

    f.momentum_derivative(t, q, &mut work.dpdt);
    *nfev += 1;
    for (pi, dpdt_i) in p.iter_mut().zip(work.dpdt.iter()) {
        *pi += (0.5 * w1) * h * *dpdt_i;
    }
}
