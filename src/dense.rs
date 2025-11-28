//! Unified dense output interpolation types.
//!
//! This module provides a safe, unified API for dense output interpolation
//! that works across all solver methods. It eliminates the need for raw pointers
//! and per-solver interpolant structs.

use crate::Float;

/// Interpolation function signature used by all solvers.
///
/// Arguments:
/// - `xi`: The point at which to interpolate
/// - `yi`: Output buffer for the interpolated state
/// - `cont`: Dense output coefficients (layout is method-specific)
/// - `xold`: Left endpoint of the step
/// - `h`: Step size (can be negative for backward integration)
pub type InterpolateFn = fn(Float, &mut [Float], &[Float], Float, Float);

/// Method-agnostic dense output interpolator for a single step.
///
/// This struct provides safe access to dense output interpolation without
/// requiring raw pointers or unsafe code. It borrows the coefficient buffer
/// and can interpolate at any point within the step interval.
///
/// # Example
/// ```ignore
/// let interpolant = StepInterpolant::new(&cont, xold, h, DOPRI5::interpolate);
/// let mut yi = vec![0.0; n];
/// interpolant.interpolate(xi, &mut yi);
/// ```
#[derive(Clone, Copy)]
pub struct StepInterpolant<'a> {
    /// Dense output coefficients (layout is method-specific)
    cont: &'a [Float],
    /// Left endpoint of the step
    xold: Float,
    /// Step size (can be negative for backward integration)
    h: Float,
    /// Interpolation function for the specific method
    interp_fn: InterpolateFn,
}

impl<'a> StepInterpolant<'a> {
    /// Create a new step interpolant.
    ///
    /// # Arguments
    /// - `cont`: Dense output coefficients computed by the solver
    /// - `xold`: Left endpoint of the step (time before the step)
    /// - `h`: Step size (positive for forward, negative for backward integration)
    /// - `interp_fn`: The interpolation function for the solver method
    #[inline]
    pub fn new(cont: &'a [Float], xold: Float, h: Float, interp_fn: InterpolateFn) -> Self {
        Self {
            cont,
            xold,
            h,
            interp_fn,
        }
    }

    /// Interpolate the solution at point `xi`.
    ///
    /// The point should ideally lie within the step interval `[xold, xold + h]`
    /// (or `[xold + h, xold]` for backward integration), but extrapolation
    /// is supported.
    #[inline]
    pub fn interpolate(&self, xi: Float, yi: &mut [Float]) {
        (self.interp_fn)(xi, yi, self.cont, self.xold, self.h)
    }

    /// Get the step bounds as `(left, right)` where `left <= right`.
    #[inline]
    pub fn bounds(&self) -> (Float, Float) {
        if self.h >= 0.0 {
            (self.xold, self.xold + self.h)
        } else {
            (self.xold + self.h, self.xold)
        }
    }

    /// Get the raw step parameters `(xold, h)`.
    #[inline]
    pub fn step_params(&self) -> (Float, Float) {
        (self.xold, self.h)
    }

    /// Convert to an owned segment for storage.
    #[inline]
    pub fn to_segment(&self) -> DenseSegment {
        DenseSegment {
            cont: self.cont.to_vec(),
            xold: self.xold,
            h: self.h,
            interp_fn: self.interp_fn,
        }
    }
}

/// Owned dense output segment for storage in `ContinuousOutput`.
///
/// This stores all the data needed to interpolate within a single step,
/// including a copy of the coefficients and the interpolation function.
#[derive(Clone)]
pub struct DenseSegment {
    /// Dense output coefficients (owned copy)
    pub cont: Vec<Float>,
    /// Left endpoint of the step
    pub xold: Float,
    /// Step size
    pub h: Float,
    /// Interpolation function
    interp_fn: InterpolateFn,
}

impl DenseSegment {
    /// Create a new owned segment.
    pub fn new(cont: Vec<Float>, xold: Float, h: Float, interp_fn: InterpolateFn) -> Self {
        Self {
            cont,
            xold,
            h,
            interp_fn,
        }
    }

    /// Interpolate the solution at point `xi`.
    #[inline]
    pub fn interpolate(&self, xi: Float, yi: &mut [Float]) {
        (self.interp_fn)(xi, yi, &self.cont, self.xold, self.h)
    }

    /// Get the step bounds as `(left, right)` where `left <= right`.
    #[inline]
    pub fn bounds(&self) -> (Float, Float) {
        if self.h >= 0.0 {
            (self.xold, self.xold + self.h)
        } else {
            (self.xold + self.h, self.xold)
        }
    }

    /// Create a borrowed interpolant from this segment.
    #[inline]
    pub fn as_interpolant(&self) -> StepInterpolant<'_> {
        StepInterpolant::new(&self.cont, self.xold, self.h, self.interp_fn)
    }
}

impl std::fmt::Debug for DenseSegment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DenseSegment")
            .field("xold", &self.xold)
            .field("h", &self.h)
            .field("cont_len", &self.cont.len())
            .finish()
    }
}
