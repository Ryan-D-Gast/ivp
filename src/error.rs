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
            Error::InvalidStepSize(v) => write!(
                f,
                "invalid step size: h = {}. h must be non-zero and its sign must match sign(xend - x)",
                v
            ),
            Error::NMaxMustBePositive(v) => write!(
                f,
                "invalid maximum step count: nmax = {} (must be > 0). Consider increasing `options.nmax` if the solver stops early",
                v
            ),
            Error::NStiffMustBePositive(v) => write!(
                f,
                "invalid stiffness test interval: nstiff = {} (must be > 0)",
                v
            ),
            Error::URoundOutOfRange(v) => write!(
                f,
                "invalid machine rounding parameter: uround = {v:.3e} (must be in (1e-35, 1.0)). Hint: for f64 use ~2.3e-16"
            ),
            Error::SafetyFactorOutOfRange(v) => {
                write!(
                    f,
                    "invalid safety factor: safety_factor = {v:.3e} (must be in (1e-4, 1.0)). Typical value: 0.9"
                )
            }
            Error::BetaTooLarge(v) => write!(
                f,
                "invalid stabilization parameter: beta = {v:.3e} (must be <= 0.2). Recommended: 0.04"
            ),
            Error::NegativeTolerance { kind, index, value } => write!(
                f,
                "{} tolerance must be non-negative at index {} (got {value:.3e}). All components of rtol/atol must be >= 0",
                kind, index
            ),
            Error::ToleranceSizeMismatch {
                kind,
                expected,
                actual,
            } => write!(
                f,
                "{} tolerance length mismatch: expected length {} (state dimension), got {}",
                kind, expected, actual
            ),
            Error::InvalidScaleFactors(min, max) => write!(
                f,
                "invalid step scaling limits: scale_min = {min:.3e}, scale_max = {max:.3e}. Require scale_min > 0 and scale_max > scale_min (typical: 0.2 and 5.0)",
            ),
            Error::DenseOutputDisabled => write!(
                f,
                "dense output is disabled. Enable `dense_output = true` in solver options to construct interpolants"
            ),
            Error::EvaluationOutOfRange(t) => {
                write!(
                    f,
                    "evaluation time {} is outside the covered interval. Ensure t lies within the solution span returned by `sol.sol_span()`",
                    t
                )
            }
            Error::NewtonMaxIterMustBePositive(v) => {
                write!(
                    f,
                    "invalid Newton iteration cap: newton_maxiter = {} (must be > 0)",
                    v
                )
            }
            Error::PivotSizeMismatch { expected, actual } => write!(
                f,
                "pivot index slice length mismatch: expected {}, got {}",
                expected, actual
            ),
            Error::SingularMatrix => write!(
                f,
                "matrix is singular. The linear system could not be solved (ill-conditioned Jacobian/mass). Try a smaller step size or provide an analytic Jacobian/mass matrix"
            ),
            Error::NonSquareMatrix { rows, cols } => {
                write!(
                    f,
                    "matrix must be square (got {} rows and {} columns)",
                    rows, cols
                )
            }
            Error::InvalidDAEPartition {
                n,
                nind1,
                nind2,
                nind3,
            } => {
                write!(
                    f,
                    "invalid DAE partition: n={}, nind1={}, nind2={}, nind3={}. Counts must be non-negative, ordered (index-1, then index-2, then index-3), and sum to n",
                    n, nind1, nind2, nind3
                )
            }
        }
    }
}

impl std::error::Error for Error {}
