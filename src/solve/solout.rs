//! Default SolOut that implements t_eval sampling and endpoint recording; wraps a user SolOut.

use crate::{
    Float,
    interpolate::Interpolate,
    ode::ODE,
    solout::{ControlFlag, SolOut},
    solve::event::{Direction, EventConfig},
};

pub(crate) struct DefaultSolOut<'a, F>
where
    F: ODE,
{
    ode: &'a F,
    t_eval: Option<Vec<Float>>,
    next_idx: usize,
    tol: Float,
    t: Vec<Float>,
    y: Vec<Vec<Float>>,
    // Dense output collection
    collect_dense: bool,
    dense_segs: Vec<(Vec<Float>, Float, Float)>, // (cont, xold, h)
    // For event handling: last state and configuration
    yold: Vec<Float>,
    // Mutable event configuration provided to the user's `event` callback
    event_config: EventConfig,
    // Cached event value from previous step (g at the previous abscissa)
    prev_event: Option<Float>,
    // Internal counter of occurred events
    event_hits: usize,
}

impl<'a, F> DefaultSolOut<'a, F>
where
    F: ODE,
{
    pub fn new(ode: &'a F, t_eval: Option<Vec<Float>>, collect_dense: bool) -> Self {
        Self {
            ode,
            t_eval,
            next_idx: 0,
            tol: 1e-12,
            t: Vec::new(),
            y: Vec::new(),
            collect_dense,
            dense_segs: Vec::new(),
            event_config: EventConfig::new(),
            prev_event: None,
            event_hits: 0,
            yold: Vec::new(),
        }
    }

    pub fn into_payload(self) -> (Vec<Float>, Vec<Vec<Float>>, Vec<(Vec<Float>, Float, Float)>) {
        (self.t, self.y, self.dense_segs)
    }
}

impl<'a, F: ODE> SolOut for DefaultSolOut<'a, F> {
    fn solout<I: Interpolate>(
        &mut self,
        xold: Float,
        x: Float,
        y: &[Float],
        interpolator: &I,
    ) -> ControlFlag {
        // Event handling: if we have a cached previous event value use it, otherwise
        // evaluate at the initial point and skip detection on this first call.
        if let Some(g_prev) = self.prev_event {
            // Evaluate event function at the current endpoint (may mutate config)
            let g_curr = self.ode.event(x, y, &mut self.event_config);

            // Helper to test if a sign change in the requested direction occurred
            let crossed = |left: Float, right: Float, dir: &Direction| -> bool {
                match dir {
                    Direction::All => {
                        (left <= 0.0 && right >= 0.0) || (left >= 0.0 && right <= 0.0)
                    }
                    Direction::Positive => left < 0.0 && right >= 0.0,
                    Direction::Negative => left > 0.0 && right <= 0.0,
                }
            };

            if crossed(g_prev, g_curr, &self.event_config.direction) {
                // Root-bracket [a,b] = [xold,x] and refine by bisection using dense output
                let mut a = xold;
                let mut b = x;
                let mut g_left = g_prev;
                let mut g_right = g_curr;
                let mut y_mid_buf = vec![0.0; y.len()];

                // Snap to an endpoint if it's already within tolerance
                if g_left.abs() <= self.tol {
                    self.t.push(a);
                    self.y.push(self.yold.clone());
                } else if g_right.abs() <= self.tol {
                    self.t.push(b);
                    self.y.push(y.to_vec());
                } else {
                    for _ in 0..100 {
                        if (b - a).abs() <= self.tol {
                            break;
                        }
                        let mid = 0.5 * (a + b);
                        interpolator.interpolate(mid, &mut y_mid_buf);
                        let g_mid = self.ode.event(mid, &y_mid_buf, &mut self.event_config);

                        // Preserve crossing direction when selecting the subinterval
                        let left_cross = crossed(g_left, g_mid, &self.event_config.direction);
                        let right_cross = crossed(g_mid, g_right, &self.event_config.direction);

                        if left_cross {
                            b = mid;
                            g_right = g_mid;
                        } else if right_cross {
                            a = mid;
                            g_left = g_mid;
                        } else {
                            // Fallback to sign-based bisection
                            if g_left.signum() * g_mid.signum() <= 0.0 {
                                b = mid;
                                g_right = g_mid;
                            } else {
                                a = mid;
                                g_left = g_mid;
                            }
                        }
                    }

                    // Record event time and state at b
                    let mut y_at_b = vec![0.0; y.len()];
                    interpolator.interpolate(b, &mut y_at_b);
                    self.t.push(b);
                    self.y.push(y_at_b);
                }

                // Count the event and check for termination
                self.event_hits += 1;
                if let Some(limit) = self.event_config.terminal_count {
                    if self.event_hits >= limit {
                        return ControlFlag::Interrupt;
                    }
                }
            }

            // Cache current event value for next step
            self.prev_event = Some(g_curr);
        } else {
            // First call: evaluate event at current point and cache it; no detection yet
            let g0 = self.ode.event(x, y, &mut self.event_config);
            self.prev_event = Some(g0);
        }

        // Update yold for next step
        self.yold = y.to_vec();

        // Optionally collect dense coefficients for this accepted step
        if self.collect_dense {
            let (cont, cxold, h) = interpolator.get_cont();
            // Skip degenerate segments
            if h != 0.0 {
                self.dense_segs.push((cont, cxold, h));
            }
        }
        // If t_eval is provided, interpolate and store values within (xold, x]
        if let Some(te) = self.t_eval.as_ref() {
            // Handle the initial call (xold == x) -> push provided y directly
            let mut i = self.next_idx;
            if (xold - x).abs() <= self.tol {
                while i < te.len() && (te[i] - x).abs() <= self.tol {
                    self.t.push(te[i]);
                    self.y.push(y.to_vec());
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
            // No t_eval: record each accepted step endpoint (including initial call)
            self.t.push(x);
            self.y.push(y.to_vec());
        }

        ControlFlag::Continue
    }
}
