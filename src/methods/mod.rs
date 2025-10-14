// Numerical methods

// --- ODE Solvers ---
mod dop853;
mod dopri5;
mod radau;
mod rk4;
mod rk23;
mod bdf;

pub use bdf::BDF;
pub use dop853::DOP853;
pub use dopri5::DOPRI5;
pub use radau::RADAU;
pub use rk4::RK4;
pub use rk23::RK23;

use crate::{Float, ode::ODE, status::Status};

use std::{
    iter::{ExactSizeIterator, FusedIterator},
    ops::{Index, IndexMut},
    slice::Iter,
};

/// --- Shared types and utilities ---

/// The output of a numerical integrator
#[derive(Clone, Debug)]
pub struct IntegrationResult {
    /// The final value of the independent variable
    pub x: Float,
    /// The step size of the next integration step
    pub h: Float,
    /// Status of the integration
    pub status: Status,
    /// The evaluation statistics
    pub evals: Evals,
    /// The step statistics
    pub steps: Steps,
}

impl IntegrationResult {
    pub fn new(x: Float, h: Float, status: Status, evals: Evals, steps: Steps) -> Self {
        Self {
            x,
            h,
            status,
            evals,
            steps,
        }
    }

    /// Returns `true` if the integration was successful completion or stopped by user request.
    pub fn is_ok(&self) -> bool {
        matches!(self.status, Status::Success | Status::UserInterrupt)
    }
}

#[derive(Clone, Debug)]
/// The calculation statistics of a numerical integrator
pub struct Evals {
    /// The number of ODE function evaluations
    pub ode: usize,
    /// The number of Jacobian evaluations
    pub jac: usize,
    /// The number of LU decompositions
    pub lu: usize,
}

impl Evals {
    pub fn new() -> Self {
        Self {
            ode: 0,
            jac: 0,
            lu: 0,
        }
    }
}

/// The step statistics of a numerical integrator
#[derive(Clone, Debug)]
pub struct Steps {
    /// Total number of steps taken
    pub total: usize,
    /// The number of accepted steps
    pub accepted: usize,
    /// The number of rejected steps
    pub rejected: usize,
}

impl Steps {
    pub fn new() -> Self {
        Self {
            total: 0,
            accepted: 0,
            rejected: 0,
        }
    }
}

/// Tolerance enum to allow scalar or vector tolerances
/// using [`Into`] trait for easy conversion from `Float`, `[Float; N]`, or `Vec<Float>`
/// users do not need to know or worry this simply allows both
/// `Float` and `[Float; N]` to be passed in as arguments.
#[derive(Clone, Debug)]
pub enum Tolerance {
    Scalar(Float),
    Vector(Vec<Float>),
}

/// Iterator over the components of a [`Tolerance`].
pub enum ToleranceIter<'a> {
    Scalar { value: Float, remaining: usize },
    Vector { iter: Iter<'a, Float> },
}

impl<'a> Iterator for ToleranceIter<'a> {
    type Item = Float;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Scalar { value, remaining } => {
                if *remaining == 0 {
                    None
                } else {
                    *remaining -= 1;
                    Some(*value)
                }
            }
            Self::Vector { iter } => iter.next().copied(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Scalar { remaining, .. } => (*remaining, Some(*remaining)),
            Self::Vector { iter } => iter.size_hint(),
        }
    }
}

impl<'a> ExactSizeIterator for ToleranceIter<'a> {}

impl<'a> FusedIterator for ToleranceIter<'a> {}

impl Tolerance {
    /// Iterate over the tolerance values for a state vector of length `len`.
    ///
    /// For scalar tolerances, the same value is repeated `len` times. Vector
    /// tolerances must match `len` exactly.
    pub fn iter(&self, len: usize) -> ToleranceIter<'_> {
        match self {
            Tolerance::Scalar(value) => ToleranceIter::Scalar {
                value: *value,
                remaining: len,
            },
            Tolerance::Vector(values) => {
                assert!(
                    values.len() == len,
                    "Tolerance vector length mismatch: expected {}, got {}",
                    len,
                    values.len()
                );
                ToleranceIter::Vector {
                    iter: values.iter(),
                }
            }
        }
    }
}

impl From<Float> for Tolerance {
    fn from(val: Float) -> Self {
        Tolerance::Scalar(val)
    }
}

impl From<&[Float]> for Tolerance {
    fn from(val: &[Float]) -> Self {
        Tolerance::Vector(val.to_vec())
    }
}

impl<const N: usize> From<[Float; N]> for Tolerance {
    fn from(val: [Float; N]) -> Self {
        Tolerance::Vector(val.to_vec())
    }
}

impl From<Vec<Float>> for Tolerance {
    fn from(val: Vec<Float>) -> Self {
        Tolerance::Vector(val)
    }
}

impl Index<usize> for Tolerance {
    type Output = Float;

    fn index(&self, index: usize) -> &Self::Output {
        match self {
            Tolerance::Scalar(v) => v,
            Tolerance::Vector(vs) => &vs[index],
        }
    }
}

impl IndexMut<usize> for Tolerance {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        match self {
            Tolerance::Scalar(v) => v,
            Tolerance::Vector(vs) => &mut vs[index],
        }
    }
}

/// Compute an initial step size guess for an ODE solver.
pub fn hinit<F>(
    f: &F,
    x: Float,
    y: &[Float],
    posneg: Float,
    f0: &[Float],
    f1: &mut [Float],
    y1: &mut [Float],
    iord: usize,
    hmax: Float,
    atol: &Tolerance,
    rtol: &Tolerance,
) -> Float
where
    F: ODE,
{
    let n = y.len();
    let mut dnf: Float = 0.0;
    let mut dny: Float = 0.0;

    for i in 0..n {
        let sk = atol[i] + rtol[i] * y[i].abs();
        dnf += (f0[i] / sk) * (f0[i] / sk);
        dny += (y[i] / sk) * (y[i] / sk);
    }

    let mut h: Float;
    if dnf <= 1e-10 || dny <= 1e-10 {
        h = 1.0e-6;
    } else {
        h = (dny / dnf).sqrt() * 0.01;
    }

    if h > hmax.abs() {
        h = hmax.abs();
    }
    h = h.abs() * posneg.signum();

    // Explicit Euler step: y1 = y + h * f0
    for i in 0..n {
        y1[i] = y[i] + h * f0[i];
    }
    // Evaluate f at x+h
    f.ode(x + h, y1, f1);

    // Estimate second derivative
    let mut der2: Float = 0.0;
    for i in 0..n {
        let sk = atol[i] + rtol[i] * y[i].abs();
        let df = (f1[i] - f0[i]) / sk;
        der2 += df * df;
    }
    der2 = der2.sqrt() / h.abs();

    let der12 = der2.abs().max(dnf.sqrt());
    let h1: Float;
    if der12 <= 1.0e-15_f64 {
        h1 = (1.0e-6_f64).max(h.abs() * 1.0e-3_f64);
    } else {
        h1 = (0.01_f64 / der12).powf(1.0_f64 / (iord as Float));
    }

    let h_final = h.abs().min(100.0_f64 * h.abs()).min(h1).min(hmax.abs());
    h_final.abs() * posneg.signum()
}