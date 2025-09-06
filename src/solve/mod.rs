//! High-level solve module: SciPy-like API pieces split into submodules.

pub mod cont;
pub mod options;
pub mod solout;
pub mod solve_ivp;
pub mod solution;

// Re-exports for ergonomic access via crate::solve::* and prelude
pub use options::{IVPOptions, Method};
pub use solution::IVPSolution;
pub use solve_ivp::solve_ivp;
