//! Matrix types, operations, and utilities.

mod add;
mod base;
mod index;
mod linear;
mod lu;
mod macros;
mod mul;
mod sub;

pub use base::{Matrix, MatrixStorage};
pub use linear::{lin_solve, lin_solve_complex};
pub use lu::{lu_decomp, lu_decomp_complex};
