//! User-supplied Initial Value Problem.

use crate::{
    Float,
    matrix::{Matrix, MatrixStorage},
    solve::event::EventConfig,
};

/// User-supplied IVP system.
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
/// impl IVP for VanDerPol {
///     fn ode(&self, x: f64, y: &[f64], dydx: &mut [f64]) {
///         dydx[0] = y[1];
///         dydx[1] = ((1.0 - y[0]*y[0])*y[1] - y[0]) / self.eps;
///     }
/// }
/// ```
pub trait IVP {
    /// Compute the derivative dydx at (x, y).
    fn ode(&self, x: Float, y: &[Float], dydx: &mut [Float]);

    /// Events to track during integration.
    ///
    /// This function is called after each successful step to check for events.
    /// The solver will find an accurate value of event(x, y) = 0 using root finding.
    /// When the event is found it will be recorded in the solution. The user can make
    /// events terminate, or the direction of zero crossing, configurable via `EventConfig`.
    #[allow(unused_variables)]
    fn event(&self, x: Float, y: &[Float], event: &mut EventConfig) -> Float {
        1.0
    }

    /// Jacobian matrix J = df/dy
    ///
    /// The jacobian matrix is a matrix of partial derivatives of a vector-valued function.
    /// It describes the local behavior of the system of equations and can be used to improve
    /// the efficiency of certain solvers by providing information about the local behavior
    /// of the system of equations.
    ///
    /// The Jacobian matrix `j` is a pre-allocated `dim x dim` matrix, where `dim` is the length of `y`.
    /// The user can fill the matrix via Index/IndexMut, e.g., `j[(row, col)] = value`.
    ///
    /// By default, this method uses a finite difference approximation.
    /// Users can override this with an analytical implementation for better efficiency.
    fn jac(&self, x: Float, y: &[Float], j: &mut Matrix) {
        debug_assert!(
            !matches!(j.storage, MatrixStorage::Banded { .. }),
            "Banded Jacobian not supported in default implementation"
        );

        // Default implementation using forward finite differences
        let dim = y.len();
        let mut y_perturbed = y.to_vec();
        let mut f_perturbed = vec![0.0; dim];
        let mut f_origin = vec![0.0; dim];

        // Compute the unperturbed derivative
        self.ode(x, y, &mut f_origin);

        // Use sqrt of machine epsilon for finite differences
        let eps = Float::EPSILON.sqrt();

        // For each column of the jacobian
        for col in 0..dim {
            // Get the original value
            let y_original_j = y[col];

            // Calculate perturbation size (max of component magnitude or 1.0)
            let perturbation = eps * y_original_j.abs().max(1.0);

            // Perturb the component
            y_perturbed[col] = y_original_j + perturbation;

            // Evaluate function with perturbed value
            self.ode(x, &y_perturbed, &mut f_perturbed);

            // Restore original value
            y_perturbed[col] = y_original_j;

            // Compute finite difference approximation for this column
            for row in 0..dim {
                j[(row, col)] = (f_perturbed[row] - f_origin[row]) / perturbation;
            }
        }
    }

    /// Mass matrix M in the system M y' = f(x, y).
    ///
    /// The mass matrix is only supported for non-explicit solvers.
    /// e.g., Radau and BDF. By default, this is the identity matrix,
    /// which results in the standard form y' = f(x, y).
    ///
    /// The mass matrix `m` is a pre-allocated `dim x dim` matrix,
    /// where `dim` is the length of `y`. The user can fill the matrix via Index/IndexMut,
    /// e.g., `m[(row, col)] = value`.
    fn mass(&self, m: &mut Matrix) {
        Matrix::identity(m.nrows());
    }
}
