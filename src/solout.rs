//! User defined callback hook executed after each accepted step.

use crate::Float;

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
/// - `cont`: dense-output coefficient table (8 coefficient vectors per state),
/// - `h`: the step size used for the accepted step.
///
/// Typical uses:
/// - print or log the solution at equidistant output points by using the
///   `contd5` helper to interpolate inside [xold, x];
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
/// impl SolOut<2> for Printer {
///     fn solout(&mut self, nstep, xold, x, y, cont, h) -> ControlFlag {
///         if nstep == 1 {
///             println!("x = {}, y = {:?}", xold, y);
///             self.xout = xold + self.dx;
///         }
///         let mut yi = y.clone();
///         while self.xout <= x {
///             contd5(cont, xold, h, self.xout, &mut yi);
///             println!("x = {}, y = {:?}", self.xout, yi);
///             self.xout += self.dx;
///         }
///         ControlFlag::Continue
///     }
/// }
/// ```
pub trait SolOut<const N: usize> {
    fn solout(
        &mut self,
        nstep: usize,
        xold: Float,
        x: Float,
        y: &[Float; N],
        cont: &[[Float; N]; 5],
        h: Float,
    ) -> ControlFlag;
}