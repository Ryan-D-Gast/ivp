//! Explicit Runge-Kutta integrators (RK23, RK4)

mod rk4;
mod rk23;

pub use rk4::rk4;
pub use rk23::rk23;

use crate::Float;

#[derive(Clone, Debug)]
pub struct RKSettings {
    pub h0: Option<Float>,
    pub hmax: Option<Float>,
    pub nmax: usize,
    pub safety_factor: Float,
    pub error_exponent: Float,
    pub scale_min: Float,
    pub scale_max: Float,
}

impl RKSettings {
    pub fn new() -> Self {
        Self {
            h0: None,
            hmax: None,
            nmax: 100_000,
            safety_factor: 0.9,
            scale_min: 0.2,
            scale_max: 5.0,
            error_exponent: -1.0 / 3.0,
        }
    }
}
