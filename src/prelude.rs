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
//!    fn derivative(&self, _x: f64, y: &[f64], _p: &[f64], dydx: &mut [f64]) {
//!       dydx[0] = y[1];
//!       dydx[1] = ((1.0 - y[0]*y[0])*y[1] - y[0]) / self.eps;
//!    }
//! }
//!
//! let vdp = VanDerPol { eps: 1e-3 };
//! let x0 = 0.0;
//! let xend = 2.0;
//! let y0 = [2.0, 0.0];
//! let t_eval = (0..=20).map(|i| i as f64 * 0.1).collect();
//! let sol = Ivp::first_order(&vdp, x0, xend, &y0)
//!     .method(Method::DOP853)
//!     .rtol(1e-6)
//!     .atol(1e-11)
//!     .t_eval(t_eval)
//!     .solve()
//!     .unwrap();
//! println!("Finished with status: {:?}", sol.status);
//! ```

pub use crate::{
    dense::{DenseSegment, StepInterpolant},
    ivp::{FirstOrderSystem, SecondOrderSystem, SeparableHamiltonianSystem},
    matrix::{Matrix, MatrixStorage},
    methods::SymplecticMethod,
    solout::ControlFlag,
    solve::event::{Direction, EventConfig},
    solve::{FirstOrderIvp, HamiltonianIvp, Ivp, JacobianSource, Method, SecondOrderIvp, Solution},
    status::Status,
};
