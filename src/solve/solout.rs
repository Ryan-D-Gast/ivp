//! Default SolOut that implements t_eval sampling and endpoint recording; wraps a user SolOut.

use crate::{
    Float,
    prelude::{Interpolate, SolOut, ControlFlag},
};

pub struct DefaultSolOut<'a, S: SolOut> {
    t_eval: Option<&'a [Float]>,
    save_endpoints: bool,
    next_idx: usize,
    tol: Float,
    t: Vec<Float>,
    y: Vec<Vec<Float>>,
    user: Option<&'a mut S>,
}

impl<'a, S: SolOut> DefaultSolOut<'a, S> {
    pub fn new(t_eval: Option<&'a [Float]>, save_endpoints: bool, user: Option<&'a mut S>) -> Self {
        Self {
            t_eval,
            save_endpoints,
            next_idx: 0,
            tol: 1e-12,
            t: Vec::new(),
            y: Vec::new(),
            user,
        }
    }

    pub fn into_data(self) -> (Vec<Float>, Vec<Vec<Float>>) {
        (self.t, self.y)
    }
}

impl<'a, S: SolOut> SolOut for DefaultSolOut<'a, S> {
    fn solout<I: Interpolate>(
        &mut self,
        xold: Float,
        x: Float,
        y: &[Float],
        interpolator: &I,
    ) -> ControlFlag {
        // Record endpoints
        if self.save_endpoints {
            if xold == x {
                self.t.push(x);
                self.y.push(y.to_vec());
            } else {
                self.t.push(x);
                self.y.push(y.to_vec());
            }
        }

        // If t_eval is provided, interpolate and store values within (xold, x]
        if let Some(te) = self.t_eval {
            // Handle the initial call (xold == x) -> only include exact match
            let mut i = self.next_idx;
            if (xold - x).abs() <= self.tol {
                while i < te.len() && (te[i] - x).abs() <= self.tol {
                    let mut yi = vec![0.0; y.len()];
                    interpolator.interpolate(te[i], &mut yi);
                    self.t.push(te[i]);
                    self.y.push(yi);
                    i += 1;
                }
            } else {
                // Regular accepted step
                // Include all te[i] in (xold, x] up to tolerance
                while i < te.len() && te[i] <= x + self.tol {
                    if te[i] >= xold - self.tol {
                        let mut yi = vec![0.0; y.len()];
                        interpolator.interpolate(te[i], &mut yi);
                        self.t.push(te[i]);
                        self.y.push(yi);
                    }
                    i += 1;
                }
            }
            self.next_idx = i;
        }

        // Forward to user callback if any
        if let Some(user) = self.user.as_deref_mut() {
            return user.solout(xold, x, y, interpolator);
        }

        ControlFlag::Continue
    }
}
