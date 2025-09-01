//! A struct representing the outputted result of a numerical integrator.

use crate::{
    Float,
    status::Status,
};

/// The output of a numerical integrator
/// 
/// This struct contains the results of the integration process,
/// including the final values of the independent and dependent
/// variables, as well as diagnostic information about the
/// integration process.
#[derive(Clone, Debug)]
pub struct Solution {
    /// The final value of the independent variable
    pub x: Float,
    /// The final value(s) of the dependent variable(s)
    pub y: Vec<Float>,
    /// The step size of the next integration step
    pub h: Float,
    /// The number of function evaluations
    pub nfev: usize,
    /// The number of steps taken
    pub nstep: usize,
    /// The number of accepted steps
    pub naccpt: usize,
    /// The number of rejected steps
    pub nrejct: usize,
    /// The status of the integration process
    pub status: Status,
}

impl Solution {
    pub fn new(x: Float, y: &[Float], h: Float, nfev: usize, nstep: usize, naccpt: usize, nrejct: usize, status: Status) -> Self {
        Self {
            x,
            y: y.to_vec(),
            h,
            nfev,
            nstep,
            naccpt,
            nrejct,
            status,
        }
    }
}