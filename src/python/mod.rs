//! Python bindings for the `ivp` crate.
//!
//! This module provides a Python interface that mimics `scipy.integrate.solve_ivp`,
//! allowing users to solve systems of ODEs using high-performance Rust solvers.
//!
//! # Submodules
//!
//! - [`solution`]: Dense output wrapper (`OdeSolution`)
//! - [`result`]: Result object (`OdeResult`)
//! - [`ivp_wrapper`]: System-trait implementations for Python callables
//! - [`solve`]: Main `solve_ivp` function
//! - [`conversion`]: Type conversion utilities
//! - [`sparsity`]: Sparse Jacobian utilities

mod conversion;
mod ivp_wrapper;
mod result;
mod solution;
mod solve;
pub mod sparsity;

use pyo3::prelude::*;
use solve::solve_ivp_py;

/// Python module registration.
#[pymodule]
pub fn ivp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_ivp_py, m)?)?;

    m.setattr(
        "__doc__",
        "A Python interface to the `ivp` Rust crate for solving initial value problems.\n\n\
         This module provides a `solve_ivp` function that mimics the interface of\n\
         `scipy.integrate.solve_ivp`, allowing users to solve systems of ODEs\n\
         using high-performance Rust solvers.\n\n\
         Supported methods:\n\
         - RK45, RK23, DOP853 (Explicit Runge-Kutta)\n\
         - Radau, BDF (Implicit methods for stiff problems)\n\
         - RK4 (Classic Runge-Kutta)\n\
         - VelocityVerlet, Ruth3, Yoshida4, SymplecticEuler* (structured symplectic methods)\n\n\
         Features:\n\
         - Dense output (continuous solution)\n\
         - Event detection (terminal and direction)\n\
         - Vectorized evaluation (optional)\n\
         - Structured symplectic integration through `solve_ivp` using either\n\
           a callable `fun(t, q)` for second-order systems, a callback pair for\n\
           Hamiltonian systems, or legacy object methods such as\n\
           `acceleration(t, q)` and `position_derivative(t, p)`/`momentum_derivative(t, q)`\n\
         - User callback validation with Python exceptions for bad shapes and\n\
           invalid callback signatures instead of uncaught Rust panics\n\
         - Argument passing to ODE functions",
    )?;

    Ok(())
}
