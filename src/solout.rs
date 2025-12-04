//! User defined callback hook executed after each accepted step.

use crate::{dense::StepInterpolant, Float};

/// Callback hook executed after each accepted step.
///
/// `SolOut` is intended for user code that wants to observe (or modify) the
/// solution as the integrator progresses. The callback is invoked once before
/// the main loop (with `nstep == 1`) and after every accepted step. The
/// arguments are:
/// - `xold`: the previous abscissa (left end of the last accepted step),
/// - `x`: the new abscissa after the accepted step (xold + h),
/// - `y`: the integrator's current solution at `x`,
/// - `interpolant`: an optional interpolator for evaluating the solution at
///   any point within the step interval [xold, x].
///
/// Typical uses:
/// - print or log the solution at equidistant output points by using the
///   `interpolant` to interpolate inside [xold, x];
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
///     fn solout(&mut self, xold, x, y, interpolant) -> ControlFlag {
///         if xold == *x {
///             println!("x = {}, y = {:?}", xold, y);
///             self.xout = xold + self.dx;
///         }
///         while self.xout <= *x {
///             if let Some(interp) = interpolant {
///                 let mut yi = vec![0.0; y.len()];
///                 interp.interpolate(self.xout, &mut yi);
///                 println!("x = {}, y = {:?}", self.xout, yi);
///             }
///             self.xout += self.dx;
///         }
///         ControlFlag::Continue
///     }
/// }
/// ```
pub trait SolOut {
    fn solout(
        &mut self,
        xold: Float,
        x: &mut Float,
        y: &mut [Float],
        interpolant: Option<&StepInterpolant<'_>>,
    ) -> ControlFlag;
}

/// Return flags for [`SolOut`].
///
/// - `Continue`: proceed with integration as normal.
/// - `Interrupt`: stop integration and return control to the caller.
/// - `ModifiedSolution`: the callback changed the solution `y` in-place; the
///   integrator will re-evaluate derivatives at the modified state before
///   continuing.
#[derive(Debug, Clone, PartialEq)]
pub enum ControlFlag {
    Continue,
    Interrupt,
    XOut(Float),
    ModifiedSolution,
}
