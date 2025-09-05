//! Default SolOut that implements t_eval sampling and endpoint recording; wraps a user SolOut.

use crate::{
    Float,
    core::{
        interpolate::Interpolate,
        solout::{ControlFlag, SolOut},
    },
};

pub(crate) struct DefaultSolOut {
    t_eval: Option<Vec<Float>>,
    next_idx: usize,
    tol: Float,
    t: Vec<Float>,
    y: Vec<Vec<Float>>,
}

impl DefaultSolOut {
    pub fn new(t_eval: Option<Vec<Float>>) -> Self {
        Self {
            t_eval,
            next_idx: 0,
            tol: 1e-12,
            t: Vec::new(),
            y: Vec::new(),
        }
    }

    pub fn into_data(self) -> (Vec<Float>, Vec<Vec<Float>>) {
        (self.t, self.y)
    }
}

impl SolOut for DefaultSolOut {
    fn solout<I: Interpolate>(
        &mut self,
        xold: Float,
        x: Float,
        y: &[Float],
        interpolator: &I,
    ) -> ControlFlag {
        // If t_eval is provided, interpolate and store values within (xold, x]
        if let Some(te) = self.t_eval.as_ref() {
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
        // If no t_eval, just store the endpoint (if not duplicate)
        } else {
            if xold == x {
                self.t.push(x);
                self.y.push(y.to_vec());
            } else {
                self.t.push(x);
                self.y.push(y.to_vec());
            }
        }

        ControlFlag::Continue
    }
}
