//! Explicit Runge-Kutta integrators (RK23, RK4)

mod rk4;
mod rk23;

pub use rk4::rk4;
pub use rk23::rk23;

use crate::Float;
use crate::status::Status;

#[derive(Clone, Debug)]
pub struct RKSettings {
    pub h0: Option<Float>,
    pub h_max: Option<Float>,
    pub nmax: usize,
}

impl RKSettings {
    pub fn new() -> Self {
        Self {
            h0: None,
            h_max: None,
            nmax: 100_000,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RKResult {
    pub x: Float,
    pub y: Vec<Float>,
    pub h: Float,
    pub nfcn: usize,
    pub nstep: usize,
    pub status: Status,
}

impl RKResult {
    pub fn new(x: Float, y: &[Float], h: Float, nfcn: usize, nstep: usize, status: Status) -> Self {
        Self {
            x,
            y: y.to_vec(),
            h,
            nfcn,
            nstep,
            status,
        }
    }
}
