//! Event Configuration and Handling

/// Configuration for event function evaluation.
pub struct EventConfig {
    /// Direction filter for zero crossings.
    pub direction: Direction,
    /// Terminate after this many event occurrences. None => never terminate.
    pub terminal_count: Option<usize>,
}

impl EventConfig {
    /// Create a new EventConfig with default settings.
    pub fn new() -> Self {
        Self {
            direction: Direction::All,
            terminal_count: None,
        }
    }

    /// Activate termination after `n` event occurrences.
    pub fn terminal_count(&mut self, n: usize) {
        self.terminal_count = Some(n);
    }

    /// Turn on termination after first event occurrence.
    pub fn terminal(&mut self) {
        self.terminal_count = Some(1);
    }

    /// Set direction of zero crossings to detect.
    pub fn direction(&mut self, dir: Direction) {
        self.direction = dir;
    }

    /// Set direction to detect all zero crossings. (Defaulted to All.)
    pub fn all(&mut self) {
        self.direction = Direction::All;
    }

    /// Set direction to only positive-going zero crossings.
    pub fn positive(&mut self) {
        self.direction = Direction::Positive;
    }

    /// Set direction to only negative-going zero crossings.
    pub fn negative(&mut self) {
        self.direction = Direction::Negative;
    }
}

pub enum Direction {
    /// Detect all zero crossings.
    All,
    /// Detect only positive-going zero crossings.
    Positive,
    /// Detect only negative-going zero crossings.
    Negative,
}

// From int
impl From<i32> for Direction {
    fn from(v: i32) -> Self {
        match v {
            x if x > 0 => Direction::Positive,
            x if x < 0 => Direction::Negative,
            _ => Direction::All,
        }
    }
}
