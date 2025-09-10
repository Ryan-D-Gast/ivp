//! High-level solve module: SciPy-like API pieces split into submodules.

pub mod cont;
pub mod options;
pub mod solout;
pub mod solution;
pub mod solve_ivp;
pub mod event;

// Required exports to use "solve_ivp"
pub use options::{Method, Options};
pub use solution::Solution;
pub use solve_ivp::solve_ivp;
