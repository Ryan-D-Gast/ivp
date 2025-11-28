//! Python bindings for the `ivp` crate.
//!
//! This module provides a Python interface that mimics `scipy.integrate.solve_ivp`,
//! allowing users to solve systems of ODEs using high-performance Rust solvers.
//!
//! # Submodules
//!
//! - [`solution`]: Dense output wrapper (`OdeSolution`)
//! - [`result`]: Result object (`OdeResult`)
//! - [`ivp_wrapper`]: IVP trait implementation for Python callables
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
         - RK4 (Classic Runge-Kutta)\n\n\
         Features:\n\
         - Dense output (continuous solution)\n\
         - Event detection (terminal and direction)\n\
         - Vectorized evaluation (optional)\n\
         - Argument passing to ODE functions",
    )?;

    Ok(())
}
