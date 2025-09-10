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
    InvalidStepSize(Float),
    InvalidScaleFactors(Float, Float),
    DenseOutputDisabled,
    EvaluationOutOfRange(Float),
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
            Error::InvalidScaleFactors(min, max) => write!(
                f,
                "scale_min must be > 0 and scale_max > scale_min (got min={}, max={})",
                min, max
            ),
            Error::DenseOutputDisabled => write!(f, "dense output is disabled"),
            Error::EvaluationOutOfRange(t) => write!(f, "evaluation time {} is outside the covered range", t),
        }
    }
}

impl std::error::Error for Error {}
