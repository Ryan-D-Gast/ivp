//! Shared types for fixed-step symplectic methods.

use crate::Float;

pub(crate) struct SymplecticWork {
    pub dqdt: Vec<Float>,
    pub dpdt: Vec<Float>,
}

impl SymplecticWork {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            dqdt: vec![0.0; n],
            dpdt: vec![0.0; n],
        }
    }
}

/// Fixed-step symplectic methods for separable Hamiltonian systems.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymplecticMethod {
    /// First-order kick-then-drift symplectic Euler.
    SymplecticEulerKickDrift,
    /// First-order drift-then-kick symplectic Euler.
    SymplecticEulerDriftKick,
    /// Second-order velocity Verlet / leapfrog.
    VelocityVerlet,
    /// Third-order Ruth composition.
    Ruth3,
    /// Fourth-order Yoshida composition of a symmetric second-order method.
    Yoshida4,
}
