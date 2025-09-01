//! A library of numerical methods for solving initial value problems (IVPs) for ordinary differential equations (ODEs).

mod ode;
mod solout;
mod status;
mod tolerance;
mod interpolate;
mod error;
mod hinit;
mod solution;

pub mod dp;
pub mod rk;

pub use ode::ODE;
pub use solution::Solution;
pub use interpolate::Interpolate;
pub use solout::{ControlFlag, SolOut};
pub use tolerance::Tolerance;
pub use error::Error;

// Prevent selecting two incompatible float precision features at once.
#[cfg(all(feature = "f32", feature = "f64"))]
compile_error!("features 'f32' and 'f64' cannot both be enabled; pick exactly one Float precision feature");

/// Change this to f128, f64, f32 as desired.
#[cfg(feature = "f32")]
pub type Float = f32;
#[cfg(feature = "f64")]
pub type Float = f64;
