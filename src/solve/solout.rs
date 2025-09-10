//! Default SolOut that implements t_eval sampling and endpoint recording; wraps a user SolOut.

use crate::{
    interpolate::Interpolate,
    ode::ODE,
    solout::{ControlFlag, SolOut},
    solve::event::EventDirection,
    Float,
};


pub(crate) struct DefaultSolOut<'a, F> 
where 
    F: ODE
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
    // For event handling
    yold: Vec<Float>,
    // Event handling configuration
    event_direction: EventDirection,
    // Terminate after this many occurrences; None => never terminate
    terminal_count: Option<usize>,
    // Internal counter of occurred events
    event_hits: usize,
}

impl<'a, F> DefaultSolOut<'a, F>
where
    F: ODE,
{
    pub fn new(ode: &'a F, t_eval: Option<Vec<Float>>, collect_dense: bool, dir: EventDirection, terminal: Option<usize>) -> Self {
        Self {
            ode,
            t_eval,
            next_idx: 0,
            tol: 1e-12,
            t: Vec::new(),
            y: Vec::new(),
            collect_dense,
            dense_segs: Vec::new(),
            event_direction: dir,
            terminal_count: terminal,
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
        // Event detection using zero crossings with direction filtering (SciPy-like)
        // Only attempt when we have a non-degenerate step
        if (x - xold).abs() > 0.0 {
            // Evaluate event function at both ends
            let g0 = self.ode.event(xold, &self.yold);
            let g1 = self.ode.event(x, y);

            // Helper to test if a sign change in desired direction occurred
            let crossed = |g_left: Float, g_right: Float, dir: EventDirection| -> bool {
                match dir {
                    EventDirection::All => (g_left <= 0.0 && g_right >= 0.0) || (g_left >= 0.0 && g_right <= 0.0),
                    EventDirection::Positive => g_left < 0.0 && g_right >= 0.0,
                    EventDirection::Negative => g_left > 0.0 && g_right <= 0.0,
                }
            };

            if crossed(g0, g1, self.event_direction) {
                // Bracket [a,b] = [xold, x] and refine by bisection using dense output
                let mut a = xold;
                let mut b = x;
                let mut ga = g0;
                let mut gb = g1;
                let mut y_mid = vec![0.0; y.len()];

                // If one endpoint already close to zero, snap to it
                if ga.abs() <= self.tol {
                    self.t.push(a);
                    self.y.push(self.yold.clone());
                } else if gb.abs() <= self.tol {
                    self.t.push(b);
                    self.y.push(y.to_vec());
                } else {
                    for _ in 0..100 {
                        if (b - a).abs() <= self.tol {
                            break;
                        }
                        let mid = 0.5 * (a + b);
                        interpolator.interpolate(mid, &mut y_mid);
                        let gm = self.ode.event(mid, &y_mid);

                        // Choose sub-interval that preserves crossing in desired direction
                        let left_cross = crossed(ga, gm, self.event_direction);
                        let right_cross = crossed(gm, gb, self.event_direction);

                        if left_cross {
                            b = mid;
                            gb = gm;
                            // y_mid becomes new right candidate; keep iterating
                        } else if right_cross {
                            a = mid;
                            ga = gm;
                        } else {
                            // Fallback: standard sign-based bisection
                            if ga.signum() * gm.signum() <= 0.0 {
                                b = mid;
                                gb = gm;
                            } else {
                                a = mid;
                                ga = gm;
                            }
                        }
                    }

                    // Use b as the event time and y at b
                    let mut yb = vec![0.0; y.len()];
                    interpolator.interpolate(b, &mut yb);
                    self.t.push(b);
                    self.y.push(yb);
                }

                // Count and decide whether to terminate
                self.event_hits += 1;
                if let Some(limit) = self.terminal_count {
                    if self.event_hits >= limit {
                        return ControlFlag::Interrupt;
                    }
                }
                // Non-terminal event: continue integration
            }
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
            // No t_eval: record each accepted step endpoint (including initial call)
            self.t.push(x);
            self.y.push(y.to_vec());
        }

        ControlFlag::Continue
    }
}
