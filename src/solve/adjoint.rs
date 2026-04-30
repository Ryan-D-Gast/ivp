//! Adjoint sensitivity analysis implementation.

use crate::{
    Float, error::Error, ivp::FirstOrderSystem, matrix::Matrix, solve::solution::Solution,
};

/// Adjoint sensitivity solver.
pub struct AdjointSolver<'a, F: FirstOrderSystem> {
    pub(crate) base: &'a F,
    pub(crate) forward_sol: &'a Solution,
}

impl<'a, F: FirstOrderSystem> AdjointSolver<'a, F> {
    pub fn new(base: &'a F, forward_sol: &'a Solution) -> Self {
        Self { base, forward_sol }
    }

    /// Solve the adjoint equations backward in time to compute gradients.
    ///
    /// # Arguments
    /// * `lambda_tf` - Terminal condition for adjoint state (d cost / dy at tf).
    /// * `dgdy` - Gradient of the running cost g(t, y, p) with respect to y.
    /// * `dgdp` - Gradient of the running cost g(t, y, p) with respect to p.
    /// * `dhdp` - Gradient of the terminal cost h(y_tf, p) with respect to p.
    pub fn compute_gradient<GY, GP, HP>(
        &self,
        p: &[Float],
        lambda_tf: &[Float],
        dgdy: GY,
        dgdp: GP,
        dhdp: HP,
    ) -> Result<Vec<Float>, Error>
    where
        GY: Fn(Float, &[Float], &[Float], &mut [Float]),
        GP: Fn(Float, &[Float], &[Float], &mut [Float]),
        HP: Fn(&[Float], &[Float], &mut [Float]),
    {
        let dim_y = lambda_tf.len();
        let dim_p = p.len();
        let (t0, tf) = self
            .forward_sol
            .sol_span()
            .ok_or(crate::error::Error::Interpolation(
                crate::error::InterpolationError::NotEnabled,
            ))?;

        // Adjoint state lambda + gradient quadrature
        // State is [lambda (dim_y), gradient (dim_p)]
        let mut y0_adj = vec![0.0; dim_y + dim_p];
        y0_adj[0..dim_y].copy_from_slice(lambda_tf);

        // Initial gradient contribution from terminal cost
        let mut dhdp_val = vec![0.0; dim_p];
        let y_tf = self.forward_sol.sol(tf)?;
        dhdp(&y_tf, p, &mut dhdp_val);
        for i in 0..dim_p {
            y0_adj[dim_y + i] = dhdp_val[i];
        }

        let adjoint_system = AdjointSensitivitySystem {
            base: self.base,
            forward_sol: self.forward_sol,
            dim_y,
            dim_p,
            dgdy,
            dgdp,
            p_params: p.to_vec(),
        };

        // Solve backward from tf to t0
        let sol = crate::solve::Ivp::first_order(&adjoint_system, tf, t0, &y0_adj).solve()?;

        let last_y = &sol.y[sol.y.len() - 1];
        let gradient = last_y[dim_y..dim_y + dim_p].to_vec();

        Ok(gradient)
    }
}

struct AdjointSensitivitySystem<'a, F: FirstOrderSystem, GY, GP> {
    base: &'a F,
    forward_sol: &'a Solution,
    dim_y: usize,
    dim_p: usize,
    dgdy: GY,
    dgdp: GP,
    p_params: Vec<Float>,
}

impl<F, GY, GP> FirstOrderSystem for AdjointSensitivitySystem<'_, F, GY, GP>
where
    F: FirstOrderSystem,
    GY: Fn(Float, &[Float], &[Float], &mut [Float]),
    GP: Fn(Float, &[Float], &[Float], &mut [Float]),
{
    fn derivative(&self, t: Float, y_adj: &[Float], _p: &[Float], dydt_adj: &mut [Float]) {
        let (lambda, _) = y_adj.split_at(self.dim_y);
        let (dlambda, dgrad) = dydt_adj.split_at_mut(self.dim_y);

        // 1. Interpolate forward solution at t and extract base state
        let y_full = self
            .forward_sol
            .sol(t)
            .unwrap_or_else(|_| vec![0.0; self.forward_sol.y[0].len()]);
        let y = &y_full[0..self.dim_y];

        // 2. Jacobian J = df/dy
        let mut j = Matrix::full(self.dim_y, self.dim_y);
        self.base.jac(t, y, &self.p_params, &mut j);

        // 3. Running cost gradient dg/dy
        let mut dgdy_val = vec![0.0; self.dim_y];
        (self.dgdy)(t, y, &self.p_params, &mut dgdy_val);

        // 4. Adjoint equation: lambda' = -J^T * lambda - (dg/dy)^T
        for i in 0..self.dim_y {
            let mut sum = -dgdy_val[i];
            for k in 0..self.dim_y {
                sum -= j[(k, i)] * lambda[k];
            }
            dlambda[i] = sum;
        }

        // 5. Gradient quadrature: grad' = -(lambda^T * df/dp + dg/dp)
        // Negated because we integrate backwards from tf to t0.
        let mut dfdp = Matrix::full(self.dim_y, self.dim_p);
        self.base
            .parameter_derivative(t, y, &self.p_params, &mut dfdp);

        let mut dgdp_val = vec![0.0; self.dim_p];
        (self.dgdp)(t, y, &self.p_params, &mut dgdp_val);

        for j_p in 0..self.dim_p {
            let mut sum = dgdp_val[j_p];
            for i_y in 0..self.dim_y {
                sum += lambda[i_y] * dfdp[(i_y, j_p)];
            }
            dgrad[j_p] = -sum;
        }
    }
}
