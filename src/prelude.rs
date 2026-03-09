//! Convenient prelude: import the most commonly used traits, types, and functions.
//!
//! # Example usage
//! ```
//! use ivp::prelude::*;
//!
//! // Van der Pol oscillator
//! struct VanDerPol { eps: f64 }
//!
//! impl FirstOrderSystem for VanDerPol {
//!    fn derivative(&self, _x: f64, y: &[f64], dydx: &mut [f64]) {
//!       dydx[0] = y[1];
//!       dydx[1] = ((1.0 - y[0]*y[0])*y[1] - y[0]) / self.eps;
//!    }
//! }
//!
//! fn main() {
//!     let vdp = VanDerPol { eps: 1e-3 };
//!     let x0 = 0.0;
//!     let xend = 2.0;
//!     let y0 = [2.0, 0.0];
//!     let t_eval = (0..=20).map(|i| i as f64 * 0.1).collect();
//!     let options = Options::builder()
//!         .method(Method::DOP853)
//!         .rtol(1e-6)
//!         .atol(1e-11)
//!         .t_eval(t_eval)
//!         .build();
//!
//!     let sol = solve_first_order_ivp(&vdp, x0, xend, &y0, options).unwrap();
//!     println!("Finished with status: {:?}", sol.status);
//! }
//! ```

pub use crate::{
    dense::{DenseSegment, StepInterpolant},
    ivp::{FirstOrderSystem, SecondOrderSystem, SeparableHamiltonianSystem},
    matrix::{Matrix, MatrixStorage},
    methods::SymplecticMethod,
    solout::ControlFlag,
    solve::event::{Direction, EventConfig},
    solve::{
        solve_first_order_ivp, solve_hamiltonian_ivp, solve_second_order_ivp, Method, Options,
        Solution, SymplecticOptions,
    },
    status::Status,
};
