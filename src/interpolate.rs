//! Interpolation trait and implementations

use crate::Float;

/// Trait for interpolating the solution within a step.
pub trait Interpolate {
    /// Interpolate the solution at the given abscissa `xi`.
    fn interpolate(&self, xi: Float, yi: &mut [Float]);
}
