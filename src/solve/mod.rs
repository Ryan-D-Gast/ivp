//! High-level solve module: SciPy-like API pieces split into submodules.

pub mod cont;
pub mod event;
pub mod first_order;
pub mod options;
pub mod solout;
pub mod solution;
pub mod symplectic;

// Required exports for the public solve APIs.
pub use first_order::solve_first_order_ivp;
pub use options::{Method, Options};
pub use solution::Solution;
pub use symplectic::{solve_hamiltonian_ivp, solve_second_order_ivp, SymplecticOptions};
