//! Errors for integration methods

use crate::Float;

/// Errors for validation of input settings
#[derive(Debug, Clone)]
pub enum Error {
    NMaxMustBePositive(usize),
    NStiffMustBePositive(usize),
    URoundOutOfRange(Float),
    SafetyFactorOutOfRange(Float),
    BetaTooLarge(Float),
    NegativeTolerance {
        kind: &'static str,
        index: usize,
        value: Float,
    },
    ToleranceSizeMismatch {
        kind: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidStepSize(Float),
    InvalidScaleFactors(Float, Float),
    DenseOutputDisabled,
    EvaluationOutOfRange(Float),
    NewtonMaxIterMustBePositive(usize),
    PivotSizeMismatch {
        expected: usize,
        actual: usize,
    },
    SingularMatrix,
    NonSquareMatrix {
        rows: usize,
        cols: usize,
    },
    InvalidDAEPartition {
        n: usize,
        nind1: usize,
        nind2: usize,
        nind3: usize,
    },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::InvalidStepSize(v) => write!(f, "step size h has invalid sign (got {})", v),
            Error::NMaxMustBePositive(v) => write!(f, "nmax must be positive (got {})", v),
            Error::NStiffMustBePositive(v) => write!(f, "nstiff must be positive (got {})", v),
            Error::URoundOutOfRange(v) => write!(f, "uround must be in (1e-35, 1.0) (got {})", v),
            Error::SafetyFactorOutOfRange(v) => {
                write!(f, "safety_factor must be in (1e-4, 1.0) (got {})", v)
            }
            Error::BetaTooLarge(v) => write!(f, "beta must be <= 0.2 (got {})", v),
            Error::NegativeTolerance { kind, index, value } => write!(
                f,
                "{} tolerance must be non-negative at index {} (got {})",
                kind, index, value
            ),
            Error::ToleranceSizeMismatch {
                kind,
                expected,
                actual,
            } => write!(
                f,
                "{} tolerance length mismatch: expected {}, got {}",
                kind, expected, actual
            ),
            Error::InvalidScaleFactors(min, max) => write!(
                f,
                "scale_min must be > 0 and scale_max > scale_min (got min={}, max={})",
                min, max
            ),
            Error::DenseOutputDisabled => write!(f, "dense output is disabled"),
            Error::EvaluationOutOfRange(t) => {
                write!(f, "evaluation time {} is outside the covered range", t)
            }
            Error::NewtonMaxIterMustBePositive(v) => {
                write!(f, "newton_maxiter must be positive (got {})", v)
            }
            Error::PivotSizeMismatch { expected, actual } => write!(
                f,
                "pivot slice size mismatch: expected {}, got {}",
                expected, actual
            ),
            Error::SingularMatrix => write!(f, "matrix is singular"),
            Error::NonSquareMatrix { rows, cols } => {
                write!(
                    f,
                    "matrix must be square (got {} rows and {} columns)",
                    rows, cols
                )
            }
            Error::InvalidDAEPartition { n, nind1, nind2, nind3 } => {
                write!(
                    f,
                    "invalid DAE partition: n={}, nind1={}, nind2={}, nind3={} (must sum to n and be ordered)",
                    n, nind1, nind2, nind3
                )
            }
        }
    }
}

impl std::error::Error for Error {}
