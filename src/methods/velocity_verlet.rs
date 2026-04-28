//! Velocity Verlet / leapfrog.

use crate::{Float, ivp::SeparableHamiltonianSystem};

use super::SymplecticWork;

pub(crate) fn step<F>(
    f: &F,
    t: Float,
    h: Float,
    q: &mut [Float],
    p: &mut [Float],
    p_params: &[Float],
    work: &mut SymplecticWork,
    nfev: &mut usize,
) where
    F: SeparableHamiltonianSystem,
{
    f.momentum_derivative(t, q, p_params, &mut work.dpdt);
    *nfev += 1;
    for (pi, dpdt_i) in p.iter_mut().zip(work.dpdt.iter()) {
        *pi += 0.5 * h * *dpdt_i;
    }

    f.position_derivative(t, p, p_params, &mut work.dqdt);
    *nfev += 1;
    for (qi, dqdt_i) in q.iter_mut().zip(work.dqdt.iter()) {
        *qi += h * *dqdt_i;
    }

    f.momentum_derivative(t, q, p_params, &mut work.dpdt);
    *nfev += 1;
    for (pi, dpdt_i) in p.iter_mut().zip(work.dpdt.iter()) {
        *pi += 0.5 * h * *dpdt_i;
    }
}
