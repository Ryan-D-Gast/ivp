//! A library of numerical methods for solving initial value problems (IVPs) for ordinary differential equations (ODEs).

mod error;
mod hinit;
mod interpolate;
mod ode;
mod solout;
mod solution;
mod status;
mod tolerance;

pub mod dp;
pub mod rk;

pub use error::Error;
pub use interpolate::Interpolate;
pub use ode::ODE;
pub use solout::{ControlFlag, SolOut};
pub use solution::Solution;
pub use tolerance::Tolerance;

// Prevent selecting two incompatible float precision features at once.
#[cfg(all(feature = "f32", feature = "f64"))]
compile_error!(
    "features 'f32' and 'f64' cannot both be enabled; pick exactly one Float precision feature"
);

/// Change this to f128, f64, f32 as desired.
#[cfg(feature = "f32")]
pub type Float = f32;
#[cfg(feature = "f64")]
pub type Float = f64;
