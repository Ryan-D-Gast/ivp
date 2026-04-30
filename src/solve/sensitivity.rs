//! Forward sensitivity analysis implementation.

use crate::{Float, ivp::FirstOrderSystem, matrix::Matrix};

/// Wrapper that augments a system with forward sensitivity equations.
///
/// For a base system y' = f(x, y, p), this wrapper integrates the augmented
/// state [y, S] where S = dy/dp is the sensitivity matrix.
/// The sensitivity equations are S' = J*S + df/dp.
pub struct ForwardSensitivitySystem<'a, F: FirstOrderSystem> {
    pub(crate) base: &'a F,
    pub(crate) dim_y: usize,
    pub(crate) dim_p: usize,
}

impl<'a, F: FirstOrderSystem> ForwardSensitivitySystem<'a, F> {
    pub fn new(base: &'a F, dim_y: usize, dim_p: usize) -> Self {
        Self { base, dim_y, dim_p }
    }
}

impl<F: FirstOrderSystem> FirstOrderSystem for ForwardSensitivitySystem<'_, F> {
    fn derivative(&self, x: Float, y_full: &[Float], p: &[Float], dydx_full: &mut [Float]) {
        let (y, s_flat) = y_full.split_at(self.dim_y);
        let (dydx, ds_flat) = dydx_full.split_at_mut(self.dim_y);

        // 1. Base derivative: y' = f(x, y, p)
        self.base.derivative(x, y, p, dydx);

        // 2. Jacobian J = df/dy (dim_y x dim_y)
        let mut j = Matrix::full(self.dim_y, self.dim_y);
        self.base.jac(x, y, p, &mut j);

        // 3. Parameter Jacobian fp = df/dp (dim_y x dim_p)
        let mut fp = Matrix::full(self.dim_y, self.dim_p);
        self.base.parameter_derivative(x, y, p, &mut fp);

        // 4. Compute S' = JS + fp
        // S is stored in column-major order in s_flat
        for col in 0..self.dim_p {
            let s_col = &s_flat[col * self.dim_y..(col + 1) * self.dim_y];
            let ds_col = &mut ds_flat[col * self.dim_y..(col + 1) * self.dim_y];

            for row in 0..self.dim_y {
                let mut sum = fp[(row, col)];
                for k in 0..self.dim_y {
                    sum += j[(row, k)] * s_col[k];
                }
                ds_col[row] = sum;
            }
        }
    }

    fn n_events(&self) -> usize {
        self.base.n_events()
    }

    fn events(&self, x: Float, y_full: &[Float], p: &[Float], out: &mut [Float]) {
        let (y, _) = y_full.split_at(self.dim_y);
        self.base.events(x, y, p, out);
    }

    fn n_quadrature(&self) -> usize {
        self.base.n_quadrature()
    }

    fn quadrature(&self, x: Float, y_full: &[Float], p: &[Float], out: &mut [Float]) {
        let (y, _) = y_full.split_at(self.dim_y);
        self.base.quadrature(x, y, p, out);
    }
}
