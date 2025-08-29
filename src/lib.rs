//! A library of numerical methods for solving initial value problems (IVPs) for ordinary differential equations (ODEs).

//mod dop853;
mod dopri5;
mod solout;
mod tolerance;
mod ode;
mod hinit;
mod settings;
mod status;
mod result;

//pub use dop853::dop853;
pub use dopri5::dopri5;
pub use solout::{SolOut, ControlFlag};
pub use tolerance::Tolerance;
pub use result::DPResult;
pub use ode::ODE;
pub use settings::DPSettings;

/// Change this to f128, f64, f32 as desired.
type Float = f64;