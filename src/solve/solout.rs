//! Default output handler for ODE solvers.
//!
//! This module provides `DefaultSolOut`, an internal implementation of the `SolOut` trait
//! that handles output sampling, event detection, and dense output collection for `solve_ivp`.

use crate::{
    Float,
    interpolate::Interpolate,
    ivp::IVP,
    solout::{ControlFlag, SolOut},
    solve::event::{Direction, EventConfig},
};

/// Internal output handler for `solve_ivp`.
pub(crate) struct DefaultSolOut<'a, F>
where
    F: IVP,
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
    t_events: Vec<Vec<Float>>,
    /// Solution states at event times
    y_events: Vec<Vec<Vec<Float>>>,
    /// Whether to collect dense output interpolation data
    collect_dense: bool,
    /// Dense output segments: (coefficients, xold, h) for each accepted step
    dense_segs: Vec<(Vec<Float>, Float, Float)>,
    /// Solution state from the previous step (for event detection)
    yold: Vec<Float>,
    /// Event detection configuration
    event_config: Vec<EventConfig>,
    /// Event function value from the previous step (for zero-crossing detection)
    prev_event: Vec<Float>,
    /// Count of detected events (for terminal event handling)
    event_hits: Vec<usize>,
    /// Optional first step size: if set, the first output after the initial condition
    /// will be at exactly `x0 + first_step` (only when `t_eval` is not provided)
    first_step: Option<Float>,
    /// Initial value of the independent variable
    x0: Float,
    /// Flag tracking whether the first-step output has been enforced
    first_output_done: bool,
    // Pre-allocated buffers for event detection (avoid per-step allocations)
    /// Buffer for current event function values
    g_curr_buf: Vec<Float>,
    /// Buffer for midpoint state during bisection
    y_mid_buf: Vec<Float>,
    /// Buffer for midpoint event values during bisection
    g_mid_buf: Vec<Float>,
}

impl<'a, F> DefaultSolOut<'a, F>
where
    F: IVP,
{
    /// Constructs a new output handler.
    pub fn new(
        ode: &'a F,
        t_eval: Option<Vec<Float>>,
        collect_dense: bool,
        first_step: Option<Float>,
        x0: Float,
        n_states: usize,
    ) -> Self {
        let n_events = ode.n_events();
        let mut event_config = Vec::with_capacity(n_events);
        for i in 0..n_events {
            event_config.push(ode.event_config(i));
        }

        Self {
            ode,
            t_eval,
            next_idx: 0,
            tol: 1e-12,
            t: Vec::new(),
            y: Vec::new(),
            t_events: vec![Vec::new(); n_events],
            y_events: vec![Vec::new(); n_events],
            collect_dense,
            dense_segs: Vec::new(),
            event_config,
            prev_event: vec![0.0; n_events],
            event_hits: vec![0; n_events],
            yold: Vec::new(),
            first_step,
            x0,
            first_output_done: false,
            // Pre-allocate buffers for event detection
            g_curr_buf: vec![0.0; n_events],
            y_mid_buf: vec![0.0; n_states],
            g_mid_buf: vec![0.0; n_events],
        }
    }

    /// Consumes the handler and returns all collected data.
    pub fn into_payload(
        self,
    ) -> (
        Vec<Float>,
        Vec<Vec<Float>>,
        Vec<Vec<Float>>,
        Vec<Vec<Vec<Float>>>,
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

impl<'a, F: IVP> SolOut for DefaultSolOut<'a, F> {
    fn solout<I: Interpolate>(
        &mut self,
        xold: Float,
        x: &mut Float,
        y: &mut [Float],
        interpolator: Option<&I>,
    ) -> ControlFlag {
        // ============================================================================
        // Dense Output Collection
        // ============================================================================
        // Collect interpolation coefficients from each accepted step for later
        // continuous evaluation. Skip the initial callback and degenerate segments.
        
        if self.collect_dense && *x != xold && interpolator.is_some() {
            let (cont, cxold, h) = interpolator.unwrap().get_cont();
            if h != 0.0 {
                self.dense_segs.push((cont, cxold, h));
            }
        }

        // ============================================================================
        // Event Detection
        // ============================================================================
        // Monitor the user-defined event function for zero-crossings. Uses bisection
        // to refine the event location when a sign change is detected.
        // 
        // Important: When multiple events occur in the same step, we must process them
        // in chronological order. If a terminal event occurs, any events after it
        // in time should not be recorded.
        
        let n_events = self.ode.n_events();
        if n_events > 0 {
            self.ode.events(*x, y, &mut self.g_curr_buf);

            // If this is the first step (yold is empty), just initialize prev_event
            if self.yold.is_empty() {
                self.prev_event.copy_from_slice(&self.g_curr_buf);
            } else {
                // Helper to check direction-aware crossing
                #[inline]
                fn crossed(left: Float, right: Float, dir: &Direction) -> bool {
                    match dir {
                        Direction::All => {
                            (left <= 0.0 && right >= 0.0) || (left >= 0.0 && right <= 0.0)
                        }
                        Direction::Positive => left < 0.0 && right >= 0.0,
                        Direction::Negative => left > 0.0 && right <= 0.0,
                    }
                }

                // First pass: find all events in this step and their refined times
                // Store as (time, event_index, y_at_event)
                let mut detected_events: Vec<(Float, usize, Vec<Float>)> = Vec::new();
                
                for i in 0..n_events {
                    let g_prev = self.prev_event[i];
                    let g_curr = self.g_curr_buf[i];
                    let config = &self.event_config[i];

                    if crossed(g_prev, g_curr, &config.direction) {
                        // Refine event location via bisection on [xold, x]
                        let mut a = xold;
                        let mut b = *x;
                        let mut g_left = g_prev;
                        let mut g_right = g_curr;

                        let (event_t, event_y) = if g_left.abs() <= self.tol {
                            (a, self.yold.clone())
                        } else if g_right.abs() <= self.tol {
                            (b, y.to_vec())
                        } else {
                            // Bisection refinement using pre-allocated buffers
                            for _ in 0..100 {
                                if (b - a).abs() <= self.tol {
                                    break;
                                }
                                let mid = 0.5 * (a + b);
                                interpolator.unwrap().interpolate(mid, &mut self.y_mid_buf);
                                self.ode.events(mid, &self.y_mid_buf, &mut self.g_mid_buf);
                                let g_mid = self.g_mid_buf[i];

                                let left_cross = crossed(g_left, g_mid, &config.direction);
                                let right_cross = crossed(g_mid, g_right, &config.direction);

                                if left_cross {
                                    b = mid;
                                    g_right = g_mid;
                                } else if right_cross {
                                    a = mid;
                                    g_left = g_mid;
                                } else {
                                    if g_left.signum() * g_mid.signum() <= 0.0 {
                                        b = mid;
                                        g_right = g_mid;
                                    } else {
                                        a = mid;
                                        g_left = g_mid;
                                    }
                                }
                            }

                            // Get final y at the refined event time
                            interpolator.unwrap().interpolate(b, &mut self.y_mid_buf);
                            (b, self.y_mid_buf.clone())
                        };

                        detected_events.push((event_t, i, event_y));
                    }
                }

                // Sort events by time (handle both forward and backward integration)
                let forward = *x > xold;
                if forward {
                    detected_events.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                } else {
                    detected_events.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                }

                // Process events in chronological order
                for (event_t, i, event_y) in detected_events {
                    let config = &self.event_config[i];
                    
                    // Record the event
                    self.t_events[i].push(event_t);
                    self.y_events[i].push(event_y.clone());
                    self.event_hits[i] += 1;

                    // Check for terminal event
                    if let Some(limit) = config.terminal_count {
                        if self.event_hits[i] >= limit {
                            // Add the terminal event point to the output
                            self.t.push(event_t);
                            self.y.push(event_y);
                            
                            // Update prev_event before returning
                            self.prev_event.copy_from_slice(&self.g_curr_buf);
                            return ControlFlag::Interrupt;
                        }
                    }
                }
                
                // Update prev_event for all events
                self.prev_event.copy_from_slice(&self.g_curr_buf);
            }
        }

        // Update state history for event detection
        if self.yold.len() != y.len() {
            self.yold = y.to_vec();
        } else {
            self.yold.copy_from_slice(y);
        }

        // ============================================================================
        // Output Sampling
        // ============================================================================
        
        if let Some(t_eval) = self.t_eval.as_ref() {
            // Mode 1: User-specified output times
            // Interpolate solution at each requested time within the current step interval.
            
            let mut i = self.next_idx;
            
            if (xold - *x).abs() <= self.tol {
                // Initial callback (xold == x): output at matching t_eval points
                while i < t_eval.len() && (t_eval[i] - *x).abs() <= self.tol {
                    self.t.push(t_eval[i]);
                    self.y.push(y.to_vec());
                    i += 1;
                }
            } else {
                // Regular accepted step: interpolate at all t_eval[i] within [xold, x] or [x, xold]
                // Handle both forward (x > xold) and backward (x < xold) integration
                let forward = *x > xold;
                
                if forward {
                    // Forward integration: t_eval[i] in (xold, x]
                    while i < t_eval.len() && t_eval[i] <= *x + self.tol {
                        if t_eval[i] >= xold - self.tol {
                            let mut yi = vec![0.0; y.len()];
                            interpolator.unwrap().interpolate(t_eval[i], &mut yi);
                            self.t.push(t_eval[i]);
                            self.y.push(yi);
                        }
                        i += 1;
                    }
                } else {
                    // Backward integration: t_eval is sorted decreasing, t_eval[i] in [x, xold)
                    while i < t_eval.len() && t_eval[i] >= *x - self.tol {
                        if t_eval[i] <= xold + self.tol {
                            let mut yi = vec![0.0; y.len()];
                            interpolator.unwrap().interpolate(t_eval[i], &mut yi);
                            self.t.push(t_eval[i]);
                            self.y.push(yi);
                        }
                        i += 1;
                    }
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
                if !self.first_output_done && (xold - *x).abs() > self.tol {
                    let target = self.x0 + h0;
                    let direction = (*x - xold).signum();
                    
                    if direction * (*x - target) >= -self.tol {
                        // We've reached or passed the target point
                        if let Some(interp) = interpolator {
                            let mut yi = vec![0.0; y.len()];
                            interp.interpolate(target, &mut yi);
                            self.t.push(target);
                            self.y.push(yi);
                            self.first_output_done = true;
                        }
                        
                        // Also output current endpoint if distinct from target
                        if (*x - target).abs() > self.tol {
                            self.t.push(*x);
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
            if self.t.is_empty() || (self.t.last().unwrap() - *x).abs() > self.tol {
                self.t.push(*x);
                self.y.push(y.to_vec());
            }
        }

        ControlFlag::Continue
    }
}
