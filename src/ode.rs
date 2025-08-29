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
/// impl ODE<2> for VanDerPol {
///     fn ode(&mut self, x: f64, y: &[f64;2], dydx: &mut [f64;2]) {
///         dydx[0] = y[1];
///         dydx[1] = ((1.0 - y[0]*y[0])*y[1] - y[0]) / self.eps;
///     }
/// }
/// ```
pub trait ODE<const N: usize> {
    fn ode(&mut self, x: Float, y: &[Float; N], dydx: &mut [Float; N]);
}
