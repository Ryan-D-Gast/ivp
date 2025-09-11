//! Event handling types for solve_ivp solvers.

/// Event zero-crossing direction filter.
/// - All: any sign change triggers.
/// - Pos: only negative -> nonnegative crossings.
/// - Neg: only positive -> nonpositive crossings.
#[derive(Copy, Clone, Debug)]
pub enum EventDirection {
    All,
    Positive,
    Negative,
}

// From int
impl From<i32> for EventDirection {
    fn from(v: i32) -> Self {
        match v {
            x if x > 0 => EventDirection::Positive,
            x if x < 0 => EventDirection::Negative,
            _ => EventDirection::All,
        }
    }
}
