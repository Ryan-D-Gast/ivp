//! Explicit Runge-Kutta integrators (RK23, RK45)

mod rk4;
mod rk23;

pub use rk4::rk4;
pub use rk23::rk23;

use crate::Float;
use crate::status::Status;

#[derive(Clone, Debug)]
pub struct RKSettings<const N: usize> {
    pub h0: Option<Float>,
    pub h_max: Option<Float>,
    pub nmax: usize,
}

impl<const N: usize> RKSettings<N> {
    pub fn new() -> Self {
        Self {
            h0: None,
            h_max: None,
            nmax: 100_000,
        }
    }
}

#[derive(Debug, Clone)]
pub enum RKInputError {
    InvalidStepSize(Float),
    Other(String),
}

impl std::fmt::Display for RKInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RKInputError::InvalidStepSize(h) => write!(f, "Invalid step size: {}", h),
            RKInputError::Other(msg) => write!(f, "Error: {}", msg),
        }
    }
}

#[derive(Clone, Debug)]
pub struct RKResult<const N: usize> {
    pub x: Float,
    pub y: [Float; N],
    pub h: Float,
    pub nfcn: usize,
    pub nstep: usize,
    pub status: Status,
}

impl<const N: usize> RKResult<N> {
    pub fn new(x: Float, y: [Float; N], h: Float, nfcn: usize, nstep: usize, status: Status) -> Self {
        Self {
            x,
            y,
            h,
            nfcn,
            nstep,
            status,
        }
    }
}
