//! ivp: Initial value problem solvers for ODEs.
//!
//! This crate provides explicit Runge–Kutta methods (RK4, RK23, DOPRI5, DOP853),
//! implicit methods (Radau, BDF), and structured fixed-step symplectic solvers
//! for separable Hamiltonian and second-order systems.
//!
//! Highlights
//! - Methods: RK4, RK23, DOPRI5, DOP853, Radau, BDF
//! - Controls: `rtol`, `atol`, `first_step`, `min_step`, `max_step`, `nmax`
//! - Sampling: internal accepted steps by default, or exact `t_eval` times
//! - Dense output: `sol(t)`, `sol_many(&ts)`, `sol_span()` on the returned `Solution`
//! - Iteration: iterate stored samples via `solution.iter()`
//! - Structured symplectic API: `solve_hamiltonian_ivp` and `solve_second_order_ivp`
//!
//! Quick start
//! ```
//! use ivp::prelude::*;
//! use std::f64::consts::PI;
//!
//! struct SHO;
//! impl FirstOrderSystem for SHO {
//!     fn derivative(&self, _x: f64, y: &[f64], dydx: &mut [f64]) {
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
//!     let xend = 2.0 * PI; // one period
//!     let y0 = [1.0, 0.0];
//!
//!     let sol = solve_first_order_ivp(&f, x0, xend, &y0, opts).unwrap();
//!
//!     // Discrete samples
//!     println!("Discrete output at accepted steps:");
//!     for (t, y) in sol.iter() {
//!         println!("x = {:>8.5}, y = {:?}", t, y);
//!     }
//!
//!     // Continuous evaluation within the solution span
//!     if let Some((t0, t1)) = sol.sol_span() {
//!         let ts = [t0, 0.5*(t0+t1), t1];
//!         let ys = sol.sol_many(&ts).unwrap();
//!         println!("\nDense output at t0, (t0+t1)/2, t1:");
//!         for (t, y) in ts.iter().zip(ys.iter()) {
//!             println!("x = {:>8.5}, y = {:?}", t, y);
//!         }
//!     }
//! }
//! ```
//!
//! Structured symplectic example
//! ```
//! use ivp::prelude::*;
//!
//! struct HarmonicOscillator;
//! impl SecondOrderSystem for HarmonicOscillator {
//!     fn acceleration(&self, _t: f64, q: &[f64], a: &mut [f64]) {
//!         a[0] = -q[0];
//!     }
//! }
//!
//! let opts = SymplecticOptions::builder()
//!     .method(SymplecticMethod::VelocityVerlet)
//!     .step_size(0.05)
//!     .build();
//!
//! let sol = solve_second_order_ivp(&HarmonicOscillator, 0.0, 1.0, &[1.0], &[0.0], opts).unwrap();
//! assert_eq!(sol.y.last().unwrap().len(), 2);
//! ```
//!
//! ## License
//!
//! ```text
//! Copyright 2025 Ryan D. Gast
//!
//! Licensed under the Apache License, Version 2.0 (the "License");
//! you may not use this file except in compliance with the License.
//! You may obtain a copy of the License at
//!
//!     http://www.apache.org/licenses/LICENSE-2.0
//!
//! Unless required by applicable law or agreed to in writing, software
//! distributed under the License is distributed on an "AS IS" BASIS,
//! WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//! See the License for the specific language governing permissions and
//! limitations under the License.
//! ```

// Feature guards
#[cfg(all(feature = "f32", feature = "f64"))]
compile_error!("Select only one of the features 'f32' or 'f64' for the Float type.");

// Numeric alias / core types
#[cfg(feature = "f32")]
pub(crate) type Float = f32;
#[cfg(feature = "f64")]
pub(crate) type Float = f64;

// -- Core modules --
pub mod dense;
pub mod error;
pub mod ivp;
pub mod matrix;
pub mod solout;
pub mod solve;
pub mod status;

#[cfg(feature = "python")]
pub mod python;

// -- Numerical methods --
pub mod methods;

// -- User convenience / re-exports --
pub mod prelude;
