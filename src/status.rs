//! Status codes for integrators

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Success,
    Interrupted,
    NeedLargerNmax,
    StepSizeTooSmall,
    ProbablyStiff,
}