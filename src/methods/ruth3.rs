//! Ruth third-order symplectic composition.

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
        *pi += h * *dpdt_i;
    }

    f.position_derivative(t, p, p_params, &mut work.dqdt);
    *nfev += 1;
    for (qi, dqdt_i) in q.iter_mut().zip(work.dqdt.iter()) {
        *qi += (-1.0 / 24.0) * h * *dqdt_i;
    }

    f.momentum_derivative(t, q, p_params, &mut work.dpdt);
    *nfev += 1;
    for (pi, dpdt_i) in p.iter_mut().zip(work.dpdt.iter()) {
        *pi += (-2.0 / 3.0) * h * *dpdt_i;
    }

    f.position_derivative(t, p, p_params, &mut work.dqdt);
    *nfev += 1;
    for (qi, dqdt_i) in q.iter_mut().zip(work.dqdt.iter()) {
        *qi += (3.0 / 4.0) * h * *dqdt_i;
    }

    f.momentum_derivative(t, q, p_params, &mut work.dpdt);
    *nfev += 1;
    for (pi, dpdt_i) in p.iter_mut().zip(work.dpdt.iter()) {
        *pi += (2.0 / 3.0) * h * *dpdt_i;
    }

    f.position_derivative(t, p, p_params, &mut work.dqdt);
    *nfev += 1;
    for (qi, dqdt_i) in q.iter_mut().zip(work.dqdt.iter()) {
        *qi += (7.0 / 24.0) * h * *dqdt_i;
    }
}
