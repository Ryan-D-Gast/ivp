//! User-supplied ODE system.

use crate::Float;

/// User-supplied ODE system.
///
/// Implement this trait for your problem to provide the right-hand side
/// function y' = f(x, y). The integrator repeatedly calls `ode` with the
/// current abscissa `x` and state `y` and expects you to fill `dydx` with the
/// derivative values.
///
/// # Example
///
/// ```ignore
/// struct VanDerPol { eps: f64 }
/// impl ODE for VanDerPol {
///     fn ode(&self, x: f64, y: &[f64], dydx: &mut [f64]) {
///         dydx[0] = y[1];
///         dydx[1] = ((1.0 - y[0]*y[0])*y[1] - y[0]) / self.eps;
///     }
/// }
/// ```
pub trait ODE {
    fn ode(&self, x: Float, y: &[Float], dydx: &mut [Float]);
}
