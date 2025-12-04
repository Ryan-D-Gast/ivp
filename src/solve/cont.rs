//! Continuous output provided by dense output coefficients (cont) from each step.

use crate::{dense::DenseSegment, Float};

use super::options::Method;

/// Piecewise dense output over all accepted steps.
#[derive(Debug, Clone)]
pub struct ContinuousOutput {
    segs: Vec<DenseSegment>,
    n_states: usize,
}

impl ContinuousOutput {
    /// Build a ContinuousOutput from per-step tuples of (cont, xold, h) and the selected method.
    pub(crate) fn from_segments(
        method: Method,
        n_states: usize,
        segs: Vec<(Vec<Float>, Float, Float)>,
    ) -> Self {
        let interp_fn = method.interpolate_fn();
        let segs = segs
            .into_iter()
            .filter(|(_, _, h)| *h != 0.0)
            .map(|(cont, xold, h)| DenseSegment::new(cont, xold, h, interp_fn))
            .collect();
        Self { segs, n_states }
    }
    
    /// Create a constant ContinuousOutput that always returns the initial state.
    /// Used for zero-interval integration (t0 == tf) and empty state vectors.
    pub(crate) fn constant(method: Method, x0: Float, y0: &[Float]) -> Self {
        let n = y0.len();
        let coeffs_per_state = method.coeffs_per_state();
        let interp_fn = method.interpolate_fn();
        
        // Create coefficients that interpolate to constant y0
        // Layout depends on method - generally first element of each block is the value
        let mut cont = vec![0.0; n * coeffs_per_state];
        
        // For RK methods: cont[0..n] = y0, derivatives in subsequent blocks
        // For BDF: cont[i * 7] = y0[i], then D1..D5 = 0, and order marker = 1
        match method {
            Method::BDF => {
                // BDF layout: for each state i, block is [D0, D1, D2, D3, D4, D5, order]
                for i in 0..n {
                    cont[i * coeffs_per_state] = y0[i];
                    // Set order marker to 1 (minimum order for constant)
                    cont[i * coeffs_per_state + coeffs_per_state - 1] = 1.0;
                }
            }
            _ => {
                // RK methods: cont[0..n] = y0, rest = 0 for zero derivatives
                cont[0..n].copy_from_slice(y0);
            }
        }
        
        // Use a tiny h to create a valid segment at x0
        let seg = DenseSegment::new(cont, x0, 1e-15, interp_fn);
        Self {
            segs: vec![seg],
            n_states: n,
        }
    }

    /// Domain covered by the dense output (inclusive on the right within tolerance).
    pub fn t_span(&self) -> Option<(Float, Float)> {
        if self.segs.is_empty() {
            return None;
        }
        let first = self.segs.first().unwrap();
        let last = self.segs.last().unwrap();
        let start = first.xold;
        let end = last.xold + last.h;
        Some((start, end))
    }

    /// Interpolate y(t) if t lies within any recorded step; returns None if outside.
    pub fn evaluate(&self, t: Float) -> Option<Vec<Float>> {
        let seg = self.find_segment(t)?;
        let mut yi = vec![0.0; self.n_states];
        seg.interpolate(t, &mut yi);
        Some(yi)
    }

    /// Batch-evaluate at many times; returns None for points outside coverage.
    pub fn evaluate_many(&self, ts: &[Float]) -> Vec<Option<Vec<Float>>> {
        ts.iter().map(|&t| self.evaluate(t)).collect()
    }

    /// Interpolate y(t) allowing extrapolation beyond the recorded range.
    /// This matches SciPy's OdeSolution.__call__ behavior.
    pub fn evaluate_extrapolate(&self, t: Float) -> Option<Vec<Float>> {
        let seg = self.find_segment_extrapolate(t)?;
        let mut yi = vec![0.0; self.n_states];
        seg.interpolate(t, &mut yi);
        Some(yi)
    }

    fn find_segment(&self, t: Float) -> Option<&DenseSegment> {
        if self.segs.is_empty() {
            return None;
        }
        
        let tol = 1e-12;
        
        // Strict interpolation - only return segment if t is within it
        for seg in &self.segs {
            let left = seg.xold.min(seg.xold + seg.h);
            let right = seg.xold.max(seg.xold + seg.h);
            if t >= left - tol && t <= right + tol {
                return Some(seg);
            }
        }
        
        None
    }
    
    fn find_segment_extrapolate(&self, t: Float) -> Option<&DenseSegment> {
        if self.segs.is_empty() {
            return None;
        }
        
        let tol = 1e-12;
        
        // First check if t is within any segment (interpolation)
        for seg in &self.segs {
            let left = seg.xold.min(seg.xold + seg.h);
            let right = seg.xold.max(seg.xold + seg.h);
            if t >= left - tol && t <= right + tol {
                return Some(seg);
            }
        }
        
        // If not within any segment, allow extrapolation using the closest segment
        // This matches SciPy's behavior
        let first = self.segs.first().unwrap();
        let last = self.segs.last().unwrap();
        
        let first_left = first.xold.min(first.xold + first.h);
        let last_right = last.xold.max(last.xold + last.h);
        
        if t < first_left {
            // Extrapolate backwards using first segment
            Some(first)
        } else if t > last_right {
            // Extrapolate forwards using last segment
            Some(last)
        } else {
            // This shouldn't happen, but return None to be safe
            None
        }
    }
}
