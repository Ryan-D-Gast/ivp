//! A library of numerical methods for solving initial value problems (IVPs) for ordinary differential equations (ODEs).

mod core;
mod error;
mod methods;

pub mod prelude;

// Prevent selecting two incompatible float precision features at once.
#[cfg(all(feature = "f32", feature = "f64"))]
compile_error!(
    "features 'f32' and 'f64' cannot both be enabled; pick exactly one Float precision feature"
);

/// Change this to f128, f64, f32 as desired.
#[cfg(feature = "f32")]
pub(crate) type Float = f32;
#[cfg(feature = "f64")]
pub(crate) type Float = f64;
