//! Status codes for integrators

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Computation is successful
    Success,
    /// Computation was successful but interrupted by user
    UserInterrupt,
    /// A larger maximum number of steps is needed
    NeedLargerNMax,
    /// Step size became too small
    StepSizeTooSmall,
    /// The problem is probably stiff
    ProbablyStiff,
    /// Too many singular matrices occurred
    SingularMatrix,
    /// Newton iteration did not converge
    PoorConvergence,
}

impl Status {
    /// Returns `true` if the integration was successful or interrupted by user.
    pub fn is_success(&self) -> bool {
        matches!(self, Status::Success | Status::UserInterrupt)
    }
}
