//! User defined callback hook executed after each accepted step.

use crate::{Float, Interpolate};

/// Callback hook executed after each accepted step.
///
/// `SolOut` is intended for user code that wants to observe (or modify) the
/// solution as the integrator progresses. The callback is invoked once before
/// the main loop (with `nstep == 1`) and after every accepted step. The
/// arguments are:
/// - `nstep`: number of accepted steps so far (starts at 1 for the initial call),
/// - `xold`: the previous abscissa (left end of the last accepted step),
/// - `x`: the new abscissa after the accepted step (xold + h),
/// - `y`: the integrator's current solution at `x`,
/// - `interpolator`: an object that can interpolate the solution between xold and x,
/// - `h`: the step size used for the accepted step.
///
/// Typical uses:
/// - print or log the solution at equidistant output points by using the
///   `interpolator` to interpolate inside [xold, x];
/// - detect events; or modify the solution in-place and return
///   `ControlFlag::ModifiedSolution` to ask the integrator to re-evaluate
///   derivatives at the changed state.
///
/// Return value:
/// - `ControlFlag::Continue` -> continue integration normally;
/// - `ControlFlag::Interrupt` -> stop integration and return to caller;
/// - `ControlFlag::ModifiedSolution` -> integrator will recompute f(x, y)
///    after the callback (the integrator expects that you updated `y`.
///
/// # Example
///
/// ```ignore
/// struct Printer {
///     xout: f64,
///     dx: f64,
/// }
/// impl SolOut for Printer {
///     fn solout(&mut self, xold, x, y, interpolator) -> ControlFlag {
///         if xold == x {
///             println!("x = {}, y = {:?}", xold, y);
///             self.xout = xold + self.dx;
///         }
///         while self.xout <= x {
///             let yi = interpolator.interpolate(self.xout);
///             println!("x = {}, y = {:?}", self.xout, yi);
///             self.xout += self.dx;
///         }
///         ControlFlag::Continue
///     }
/// }
/// ```
pub trait SolOut {
    fn solout<I: Interpolate>(
        &mut self,
        xold: Float,
        x: Float,
        y: &[Float],
        interpolator: &I,
    ) -> ControlFlag;
}

/// Return flags for [`SolOut`].
///
/// - `Continue`: proceed with integration as normal.
/// - `Interrupt`: stop integration and return control to the caller.
/// - `ModifiedSolution`: the callback changed the solution `y` in-place; the
///   integrator will re-evaluate derivatives at the modified state before
///   continuing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlFlag {
    Continue,
    Interrupt,
    ModifiedSolution,
}

/// Dummy implementation of `SolOut` that does nothing.
pub struct DummySolOut;

impl SolOut for DummySolOut {
    fn solout<I: crate::interpolate::Interpolate>(
        &mut self,
        _xold: Float,
        _x: Float,
        _y: &[Float],
        _interpolator: &I,
    ) -> crate::ControlFlag {
        crate::ControlFlag::Continue
    }
}