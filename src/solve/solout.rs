//! Default output handler for ODE solvers.
//!
//! This module provides `DefaultSolOut`, an internal implementation of the `SolOut` trait
//! that handles output sampling, event detection, and dense output collection for `solve_ivp`.

use crate::{
    Float,
    interpolate::Interpolate,
    ode::ODE,
    solout::{ControlFlag, SolOut},
    solve::event::{Direction, EventConfig},
};

/// Internal output handler for `solve_ivp`.
pub(crate) struct DefaultSolOut<'a, F>
where
    F: ODE,
{
    /// Reference to the ODE system
    ode: &'a F,
    /// User-specified output times (if provided)
    t_eval: Option<Vec<Float>>,
    /// Current index into `t_eval` for tracking progress
    next_idx: usize,
    /// Numerical tolerance for time comparisons
    tol: Float,
    /// Collected output times
    t: Vec<Float>,
    /// Collected solution states corresponding to `t`
    y: Vec<Vec<Float>>,
    /// Times at which events occurred
    t_events: Vec<Float>,
    /// Solution states at event times
    y_events: Vec<Vec<Float>>,
    /// Whether to collect dense output interpolation data
    collect_dense: bool,
    /// Dense output segments: (coefficients, xold, h) for each accepted step
    dense_segs: Vec<(Vec<Float>, Float, Float)>,
    /// Solution state from the previous step (for event detection)
    yold: Vec<Float>,
    /// Event detection configuration
    event_config: EventConfig,
    /// Event function value from the previous step (for zero-crossing detection)
    prev_event: Option<Float>,
    /// Count of detected events (for terminal event handling)
    event_hits: usize,
    /// Optional first step size: if set, the first output after the initial condition
    /// will be at exactly `x0 + first_step` (only when `t_eval` is not provided)
    first_step: Option<Float>,
    /// Initial value of the independent variable
    x0: Float,
    /// Flag tracking whether the first-step output has been enforced
    first_output_done: bool,
}

impl<'a, F> DefaultSolOut<'a, F>
where
    F: ODE,
{
    /// Constructs a new output handler.
    pub fn new(
        ode: &'a F,
        t_eval: Option<Vec<Float>>,
        collect_dense: bool,
        first_step: Option<Float>,
        x0: Float,
    ) -> Self {
        Self {
            ode,
            t_eval,
            next_idx: 0,
            tol: 1e-12,
            t: Vec::new(),
            y: Vec::new(),
            t_events: Vec::new(),
            y_events: Vec::new(),
            collect_dense,
            dense_segs: Vec::new(),
            event_config: EventConfig::new(),
            prev_event: None,
            event_hits: 0,
            yold: Vec::new(),
            first_step,
            x0,
            first_output_done: false,
        }
    }

    /// Consumes the handler and returns all collected data.
    pub fn into_payload(
        self,
    ) -> (
        Vec<Float>,
        Vec<Vec<Float>>,
        Vec<Float>,
        Vec<Vec<Float>>,
        Vec<(Vec<Float>, Float, Float)>,
    ) {
        (
            self.t,
            self.y,
            self.t_events,
            self.y_events,
            self.dense_segs,
        )
    }
}

impl<'a, F: ODE> SolOut for DefaultSolOut<'a, F> {
    fn solout<I: Interpolate>(
        &mut self,
        xold: Float,
        x: Float,
        y: &[Float],
        interpolator: Option<&I>,
    ) -> ControlFlag {
        // ============================================================================
        // Event Detection
        // ============================================================================
        // Monitor the user-defined event function for zero-crossings. Uses bisection
        // to refine the event location when a sign change is detected.
        
        if let Some(g_prev) = self.prev_event {
            // Evaluate event function at the current endpoint
            let g_curr = self.ode.event(x, y, &mut self.event_config);

            // Check if a zero-crossing occurred in the requested direction
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
                // Refine event location via bisection on [xold, x]
                let mut a = xold;
                let mut b = x;
                let mut g_left = g_prev;
                let mut g_right = g_curr;
                let mut y_mid_buf = vec![0.0; y.len()];

                // Check if event is already at an endpoint (within tolerance)
                if g_left.abs() <= self.tol {
                    self.t_events.push(a);
                    self.y_events.push(self.yold.clone());
                } else if g_right.abs() <= self.tol {
                    self.t_events.push(b);
                    self.y_events.push(y.to_vec());
                } else {
                    // Bisection refinement
                    for _ in 0..100 {
                        if (b - a).abs() <= self.tol {
                            break;
                        }
                        let mid = 0.5 * (a + b);
                        interpolator.unwrap().interpolate(mid, &mut y_mid_buf);
                        let g_mid = self.ode.event(mid, &y_mid_buf, &mut self.event_config);

                        // Select subinterval preserving the zero-crossing
                        let left_cross = crossed(g_left, g_mid, &self.event_config.direction);
                        let right_cross = crossed(g_mid, g_right, &self.event_config.direction);

                        if left_cross {
                            b = mid;
                            g_right = g_mid;
                        } else if right_cross {
                            a = mid;
                            g_left = g_mid;
                        } else {
                            // Fallback: standard sign-based bisection
                            if g_left.signum() * g_mid.signum() <= 0.0 {
                                b = mid;
                                g_right = g_mid;
                            } else {
                                a = mid;
                                g_left = g_mid;
                            }
                        }
                    }

                    // Record refined event location
                    let mut y_at_b = vec![0.0; y.len()];
                    interpolator.unwrap().interpolate(b, &mut y_at_b);
                    self.t_events.push(b);
                    self.y_events.push(y_at_b);
                }

                // Check for terminal event
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
            // Initial call: evaluate and cache event value without detecting crossings
            let g0 = self.ode.event(x, y, &mut self.event_config);
            self.prev_event = Some(g0);
        }

        // Update state history for event detection
        self.yold = y.to_vec();

        // ============================================================================
        // Dense Output Collection
        // ============================================================================
        // Collect interpolation coefficients from each accepted step for later
        // continuous evaluation. Skip the initial callback and degenerate segments.
        
        if self.collect_dense && x != xold && interpolator.is_some() {
            let (cont, cxold, h) = interpolator.unwrap().get_cont();
            if h != 0.0 {
                self.dense_segs.push((cont, cxold, h));
            }
        }

        // ============================================================================
        // Output Sampling
        // ============================================================================
        
        if let Some(t_eval) = self.t_eval.as_ref() {
            // Mode 1: User-specified output times
            // Interpolate solution at each requested time within the current step interval.
            
            let mut i = self.next_idx;
            
            if (xold - x).abs() <= self.tol {
                // Initial callback (xold == x): output at matching t_eval points
                while i < t_eval.len() && (t_eval[i] - x).abs() <= self.tol {
                    self.t.push(t_eval[i]);
                    self.y.push(y.to_vec());
                    i += 1;
                }
            } else {
                // Regular accepted step: interpolate at all t_eval[i] in (xold, x]
                while i < t_eval.len() && t_eval[i] <= x + self.tol {
                    if t_eval[i] >= xold - self.tol {
                        let mut yi = vec![0.0; y.len()];
                        interpolator.unwrap().interpolate(t_eval[i], &mut yi);
                        self.t.push(t_eval[i]);
                        self.y.push(yi);
                    }
                    i += 1;
                }
            }
            self.next_idx = i;
        } else {
            // Mode 2: Solver-selected output times
            // Record accepted step endpoints. If first_step is set, enforce that the
            // first output (after the initial condition) occurs at exactly x0 + first_step.
            
            if let Some(h0) = self.first_step {
                // First-step enforcement: skip intermediate outputs until we reach/pass
                // the target, then interpolate to the exact point.
                if !self.first_output_done && (xold - x).abs() > self.tol {
                    let target = self.x0 + h0;
                    let direction = (x - xold).signum();
                    
                    if direction * (x - target) >= -self.tol {
                        // We've reached or passed the target point
                        if let Some(interp) = interpolator {
                            let mut yi = vec![0.0; y.len()];
                            interp.interpolate(target, &mut yi);
                            self.t.push(target);
                            self.y.push(yi);
                            self.first_output_done = true;
                        }
                        
                        // Also output current endpoint if distinct from target
                        if (x - target).abs() > self.tol {
                            self.t.push(x);
                            self.y.push(y.to_vec());
                        }
                        return ControlFlag::Continue;
                    } else {
                        // Haven't reached target yet; skip this output
                        return ControlFlag::Continue;
                    }
                }
            }
            
            // Normal output: record endpoint (avoid duplicates)
            if self.t.is_empty() || (self.t.last().unwrap() - x).abs() > self.tol {
                self.t.push(x);
                self.y.push(y.to_vec());
            }
        }

        ControlFlag::Continue
    }
}
