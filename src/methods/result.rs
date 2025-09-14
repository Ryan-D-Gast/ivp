//! A struct representing the outputted result of a numerical integrator.

use crate::{Float, status::Status};

/// The output of a numerical integrator
#[derive(Clone, Debug)]
pub struct IntegrationResult {
    /// The final value of the independent variable
    pub x: Float,
    /// The final value(s) of the dependent variable(s)
    pub y: Vec<Float>,
    /// The step size of the next integration step
    pub h: Float,
    /// The number of function evaluations
    pub nfev: usize,
    /// The number of Jacobian evaluations
    pub njev: usize,
    /// The number of linear system (Ax = b)
    pub nsol: usize,
    /// The number of LU decompositions
    pub ndec: usize,
    /// The number of steps taken
    pub nstep: usize,
    /// The number of accepted steps
    pub naccpt: usize,
    /// The number of rejected steps
    pub nrejct: usize,
    /// The status of the integration process
    pub status: Status,
}

impl IntegrationResult {
    pub fn new(
        x: Float,
        y: Vec<Float>,
        h: Float,
        nfev: usize,
        njev: usize,
        nsol: usize,
        ndec: usize,
        nstep: usize,
        naccpt: usize,
        nrejct: usize,
        status: Status,
    ) -> Self {
        Self {
            x,
            y,
            h,
            nfev,
            njev,
            nsol,
            ndec,
            nstep,
            naccpt,
            nrejct,
            status,
        }
    }
}
