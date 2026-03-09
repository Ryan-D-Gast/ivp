//! Symplectic Euler variants.

use crate::{ivp::SeparableHamiltonianSystem, Float};

use super::SymplecticWork;

pub(crate) fn kick_drift_step<F>(
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
    f.momentum_derivative(t, q, &mut work.dpdt);
    *nfev += 1;
    for (pi, dpdt_i) in p.iter_mut().zip(work.dpdt.iter()) {
        *pi += h * *dpdt_i;
    }

    f.position_derivative(t, p, &mut work.dqdt);
    *nfev += 1;
    for (qi, dqdt_i) in q.iter_mut().zip(work.dqdt.iter()) {
        *qi += h * *dqdt_i;
    }
}

pub(crate) fn drift_kick_step<F>(
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
    f.position_derivative(t, p, &mut work.dqdt);
    *nfev += 1;
    for (qi, dqdt_i) in q.iter_mut().zip(work.dqdt.iter()) {
        *qi += h * *dqdt_i;
    }

    f.momentum_derivative(t, q, &mut work.dpdt);
    *nfev += 1;
    for (pi, dpdt_i) in p.iter_mut().zip(work.dpdt.iter()) {
        *pi += h * *dpdt_i;
    }
}
