//! ivp: Initial value problem solvers for ODEs.
//!
//! This crate provides explicit Runge–Kutta methods (RK4, RK23, DOPRI5, DOP853) with
//! adaptive step size control, optional dense output (continuous interpolation),
//! and a convenient solution type with both discrete samples and interpolation helpers.
//!
//! Highlights
//! - Methods: RK4 (fixed step), RK23, DOPRI5, DOP853 (adaptive)
//! - Controls: `rtol`, `atol`, `first_step`, `min_step`, `max_step`, `nmax`
//! - Sampling: internal accepted steps by default, or exact `t_eval` times
//! - Dense output: `sol(t)`, `sol_many(&ts)`, `sol_span()` on the returned `Solution`
//! - Iteration: iterate stored samples via `solution.iter()`
//!
//! Quick start
//! ```rust,no_run
//! use ivp::prelude::*;
//!
//! struct SHO;
//! impl ODE for SHO {
//!     fn ode(&self, _x: f64, y: &[f64], dydx: &mut [f64]) {
//!         dydx[0] = y[1];
//!         dydx[1] = -y[0];
//!     }
//! }
//!
//! fn main() {
//!     let opts = Options::builder()
//!         .method(Method::DOP853)
//!         .rtol(1e-9).atol(1e-9)
//!         .dense_output(true)
//!         .build();
//! 
//!     let f = SHO;
//!     let x0 = 0.0;
//!     let xend = 2.0 * std::f64::consts::PI; // one period
//!     let y0 = [1.0, 0.0];
//!
//!     let sol = solve_ivp(&f, x0, xend, &y0, opts).unwrap();
//!
//!     // Discrete samples
//!     for (t, y) in sol.iter() {
//!         // use t and y (slice)
//!     }
//!
//!     // Continuous evaluation within the solution span
//!     if let Some((t0, t1)) = sol.sol_span() {
//!         let ts = [t0, 0.5*(t0+t1), t1];
//!         let ys = sol.sol_many(&ts); // Vec<Option<Vec<f64>>>
//!         // If dense output was disabled, ys will be [None, None, None]
//!     }
//! }
//! ```
//!
//! See the examples folder for more usage patterns.

mod core;
mod error;
mod solve;

pub mod methods;
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
