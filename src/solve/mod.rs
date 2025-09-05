//! High-level solve module: SciPy-like API pieces split into submodules.

pub mod options;
pub mod solout;
pub mod solve_ivp;

// Re-exports for ergonomic access via crate::solve::* and prelude
pub use options::{IVPOptions, Method};
pub use solve_ivp::{IVPSolution, solve_ivp};
