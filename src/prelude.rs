//! Convenient prelude: import the most commonly used traits, types, and functions.
//!
//! Bring this into scope with:
//!
//! ```rust
//! use ivp::prelude::*;
//! ```
//!
//! Re-exports included:
//! - Core traits and types: `ODE`, `Interpolate`, `SolOut`, `ControlFlag`, `IntegrationResult`, `Status`.
//! - High-level API: `solve_ivp`, `Options`, `Solution`, and `Method`.
//!

pub use crate::{
    interpolate::Interpolate,
    ode::ODE,
    solout::{ControlFlag, SolOut},
    status::Status,
};
pub use crate::solve::{Options, Solution, Method, solve_ivp};
