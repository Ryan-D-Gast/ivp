//! A struct representing the outputted result of a numerical integrator.

use crate::{
    Float,
    status::Status,
};

#[derive(Clone, Debug)]
pub struct Solution {
    pub x: Float,
    pub y: Vec<Float>,
    pub h: Float,
    pub nfev: usize,
    pub nstep: usize,
    pub naccpt: usize,
    pub nrejct: usize,
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