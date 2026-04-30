//! High-level solve module: SciPy-like API pieces split into submodules.

pub mod adjoint;
mod builder;
pub mod cont;
pub mod event;
mod first_order;
mod options;
pub mod sensitivity;
mod solout;
pub mod solution;
mod symplectic;

// Required exports for the public solve APIs.
pub use builder::{FirstOrderIvp, HamiltonianIvp, Ivp, SecondOrderIvp};
pub use options::{JacobianSource, Method};
pub use solution::Solution;

pub(crate) use first_order::solve_first_order_impl;
pub(crate) use symplectic::{SymplecticConfig, solve_hamiltonian_impl, solve_second_order_impl};
