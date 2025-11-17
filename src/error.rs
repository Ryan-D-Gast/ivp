//! Errors for integration methods

use crate::Float;

/// Top-level error type for IVP solver operations
#[derive(Debug, Clone)]
pub enum Error {
    /// Configuration validation failed with one or more issues
    Config(ConfigError),
    /// Linear algebra operation failed
    LinearAlgebra(LinearAlgebraError),
    /// Dense output/interpolation operation failed
    Interpolation(InterpolationError),
}

/// Configuration validation errors for solver parameters
#[derive(Debug, Clone)]
pub enum ConfigError {
    /// Parameter must be positive
    MustBePositive {
        parameter: &'static str,
        value: usize,
    },
    /// Floating-point parameter out of valid range
    OutOfRange {
        parameter: &'static str,
        value: Float,
        min: Float,
        max: Float,
    },
    /// Tolerance component is negative
    NegativeTolerance {
        kind: &'static str,
        index: usize,
        value: Float,
    },
    /// Tolerance vector length doesn't match state dimension
    ToleranceSizeMismatch {
        kind: &'static str,
        expected: usize,
        actual: usize,
    },
    /// Step size is invalid (zero or wrong sign)
    InvalidStepSize {
        value: Float,
        expected_sign: Float,
    },
    /// Step scaling factors are invalid
    InvalidScaleFactors {
        min: Float,
        max: Float,
    },
    /// DAE index partition is invalid
    InvalidDAEPartition {
        n: usize,
        nind1: usize,
        nind2: usize,
        nind3: usize,
    },
}

/// Linear algebra errors
#[derive(Debug, Clone)]
pub enum LinearAlgebraError {
    /// Matrix is singular (determinant is zero)
    SingularMatrix,
    /// Matrix must be square for this operation
    NonSquareMatrix { rows: usize, cols: usize },
    /// Pivot array size doesn't match matrix dimension
    PivotSizeMismatch { expected: usize, actual: usize },
}

/// Dense output and interpolation errors
#[derive(Debug, Clone)]
pub enum InterpolationError {
    /// Dense output was not enabled in solver options
    NotEnabled,
    /// Evaluation point is outside the solution span
    OutOfRange { t: Float, t_start: Float, t_end: Float },
}

impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConfigError::MustBePositive { parameter, value } => {
                write!(
                    f,
                    "invalid {}: {} (must be > 0). Consider increasing this parameter if needed",
                    parameter, value
                )
            }
            ConfigError::OutOfRange { parameter, value, min, max } => {
                write!(
                    f,
                    "invalid {}: {:.3e} (must be in ({:.3e}, {:.3e}))",
                    parameter, value, min, max
                )
            }
            ConfigError::NegativeTolerance { kind, index, value } => {
                write!(
                    f,
                    "{} tolerance must be non-negative at index {} (got {:.3e}). All components of rtol/atol must be >= 0",
                    kind, index, value
                )
            }
            ConfigError::ToleranceSizeMismatch { kind, expected, actual } => {
                write!(
                    f,
                    "{} tolerance length mismatch: expected {} (state dimension), got {}",
                    kind, expected, actual
                )
            }
            ConfigError::InvalidStepSize { value, expected_sign } => {
                write!(
                    f,
                    "invalid step size: h = {}. h must be non-zero and its sign must match sign(xend - x) = {}",
                    value, expected_sign.signum()
                )
            }
            ConfigError::InvalidScaleFactors { min, max } => {
                write!(
                    f,
                    "invalid step scaling limits: scale_min = {:.3e}, scale_max = {:.3e}. Require scale_min > 0 and scale_max > scale_min (typical: 0.2 and 5.0)",
                    min, max
                )
            }
            ConfigError::InvalidDAEPartition { n, nind1, nind2, nind3 } => {
                write!(
                    f,
                    "invalid DAE partition: n={}, nind1={}, nind2={}, nind3={}. Counts must be non-negative, ordered (index-1, then index-2, then index-3), and sum to n",
                    n, nind1, nind2, nind3
                )
            }
        }
    }
}

impl std::fmt::Display for LinearAlgebraError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LinearAlgebraError::SingularMatrix => {
                write!(
                    f,
                    "matrix is singular. The linear system could not be solved (ill-conditioned Jacobian/mass). Try a smaller step size or provide an analytic Jacobian/mass matrix"
                )
            }
            LinearAlgebraError::NonSquareMatrix { rows, cols } => {
                write!(
                    f,
                    "matrix must be square (got {} rows and {} columns)",
                    rows, cols
                )
            }
            LinearAlgebraError::PivotSizeMismatch { expected, actual } => {
                write!(
                    f,
                    "pivot index slice length mismatch: expected {}, got {}",
                    expected, actual
                )
            }
        }
    }
}

impl std::fmt::Display for InterpolationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InterpolationError::NotEnabled => {
                write!(
                    f,
                    "dense output is disabled. Enable `dense_output = true` in solver options to construct interpolants"
                )
            }
            InterpolationError::OutOfRange { t, t_start, t_end } => {
                write!(
                    f,
                    "evaluation time {} is outside the covered interval [{}, {}]. Ensure t lies within the solution span returned by `sol.sol_span()`",
                    t, t_start, t_end
                )
            }
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Config(e) => write!(f, "configuration error: {}", e),
            Error::LinearAlgebra(e) => write!(f, "linear algebra error: {}", e),
            Error::Interpolation(e) => write!(f, "interpolation error: {}", e),
        }
    }
}

impl std::error::Error for Error {}
impl std::error::Error for ConfigError {}
impl std::error::Error for LinearAlgebraError {}
impl std::error::Error for InterpolationError {}
