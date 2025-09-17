//! A struct representing the outputted result of a numerical integrator.

use crate::{Float, status::Status};

#[derive(Clone, Debug)]
/// The calculation statistics of a numerical integrator
pub struct Evals {
    /// The number of ODE function evaluations
    pub ode: usize,
    /// The number of Jacobian evaluations
    pub jac: usize,
    /// The number of LU decompositions
    pub lu: usize,
}

impl Evals {
    pub fn new() -> Self {
        Self { ode: 0, jac: 0, lu: 0 }
    }
}

/// The step statistics of a numerical integrator
#[derive(Clone, Debug)]
pub struct Steps {
    /// Total number of steps taken
    pub total: usize,
    /// The number of accepted steps
    pub accepted: usize,
    /// The number of rejected steps
    pub rejected: usize,
}

impl Steps {
    pub fn new() -> Self {
        Self { total: 0, accepted: 0, rejected: 0 }
    }
}

/// The output of a numerical integrator
#[derive(Clone, Debug)]
pub struct IntegrationResult {
    /// The final value of the independent variable
    pub x: Float,
    /// The step size of the next integration step
    pub h: Float,
    /// Status of the integration
    pub status: Status,
    /// The evaluation statistics
    pub evals: Evals,
    /// The step statistics
    pub steps: Steps,
}

impl IntegrationResult {
    pub fn new(
        x: Float,
        h: Float,
        status: Status,
        evals: Evals,
        steps: Steps,
    ) -> Self {
        Self {
            x,
            h,
            status,
            evals,
            steps,
        }
    }
}
