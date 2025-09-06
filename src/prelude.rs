//! Convenient prelude: import the most commonly used traits, types, and functions.
//!
//! Bring this into scope with:
//!
//! ```rust
//! use ivp::prelude::*;
//! ```
//!
//! Re-exports included:
//! - Core traits and types: `ODE`, `Interpolate`, `SolOut`, `ControlFlag`, `Solution`, `Status`.
//! - High-level API: `solve_ivp`, `IVPOptions`, `IVPSolution`, and `Method`.
//!

pub use crate::core::{
    interpolate::Interpolate,
    ode::ODE,
    solout::{ControlFlag, SolOut},
    solution::Solution,
    status::Status,
};
pub use crate::solve::{IVPOptions, IVPSolution, Method, solve_ivp};
