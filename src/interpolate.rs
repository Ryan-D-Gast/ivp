//! Share interpolation implementations

use crate::Float;

/// Trait for interpolating the solution within a step.
pub trait Interpolate {
    /// Interpolate the solution at the given abscissa `xi`.
    fn interpolate(&self, xi: Float, yi: &mut [Float]);
}

pub struct CubicHermite<'a> {
    x0: &'a Float,
    h: &'a Float,
    y0: &'a [Float],
    y1: &'a [Float],
    dy0: &'a [Float],
    dy1: &'a [Float],
}

impl<'a> CubicHermite<'a> {
    pub fn new(
        x0: &'a Float,
        h: &'a Float,
        y0: &'a [Float],
        y1: &'a [Float],
        dy0: &'a [Float],
        dy1: &'a [Float],
    ) -> Self {
        Self {
            x0,
            h,
            y0,
            y1,
            dy0,
            dy1,
        }
    }
}

impl<'a> Interpolate for CubicHermite<'a> {
    fn interpolate(&self, xi: Float, yi: &mut [Float]) {
        // Cubic Hermite interpolation
        let t = (xi - self.x0) / self.h;
        let t2 = t * t;
        let t3 = t2 * t;

        let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
        let h10 = t3 - 2.0 * t2 + t;
        let h01 = -2.0 * t3 + 3.0 * t2;
        let h11 = t3 - t2;

        for i in 0..self.y0.len() {
            yi[i] = h00 * self.y0[i]
                + h10 * self.h * self.dy0[i]
                + h01 * self.y1[i]
                + h11 * self.h * self.dy1[i];
        }
    }
}
