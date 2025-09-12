//! Continuous output provided by dense output coefficients (cont) from each step.

use crate::{
    Float,
    methods::{
        dp::{contdp5, contdp8},
        rk::{contrk4, contrk23},
        radau::contr5,
    },
};

use super::options::Method;

type ContFn = fn(Float, &mut [Float], &[Float], Float, Float);

#[derive(Debug, Clone)]
struct Segment {
    cont: Vec<Float>,
    xold: Float,
    h: Float,
}

/// Piecewise dense output over all accepted steps.
#[derive(Debug, Clone)]
pub struct ContinuousOutput {
    segs: Vec<Segment>,
    cont_fn: ContFn,
    coeffs_per_state: usize,
}

impl ContinuousOutput {
    /// Build a ContinuousOutput from per-step tuples of (cont, xold, h) and the selected method.
    pub(crate) fn from_segments(method: Method, segs: Vec<(Vec<Float>, Float, Float)>) -> Self {
        let (cont_fn, coeffs_per_state) = match method {
            Method::RK4 => (contrk4 as ContFn, 4),
            Method::RK23 => (contrk23 as ContFn, 4),
            Method::DOPRI5 => (contdp5 as ContFn, 5),
            Method::DOP853 => (contdp8 as ContFn, 8),
            Method::Radau5 => (contr5 as ContFn, 4),
        };
        let segs = segs
            .into_iter()
            .filter(|(_, _, h)| *h != 0.0)
            .map(|(cont, xold, h)| Segment { cont, xold, h })
            .collect();
        Self {
            segs,
            cont_fn,
            coeffs_per_state,
        }
    }

    /// Domain covered by the dense output (inclusive on the right within tolerance).
    pub fn t_span(&self) -> Option<(Float, Float)> {
        if self.segs.is_empty() {
            return None;
        }
        let start = self.segs.first().unwrap().xold;
        let end = self.segs.last().map(|s| s.xold + s.h).unwrap();
        Some((start, end))
    }

    /// Interpolate y(t) if t lies within any recorded step; returns None if outside.
    pub fn evaluate(&self, t: Float) -> Option<Vec<Float>> {
        let seg = self.find_segment(t)?;
        let n = seg.cont.len() / self.coeffs_per_state;
        let mut yi = vec![0.0; n];
        (self.cont_fn)(t, &mut yi, &seg.cont, seg.xold, seg.h);
        Some(yi)
    }

    /// Batch-evaluate at many times; returns None for points outside coverage.
    pub fn evaluate_many(&self, ts: &[Float]) -> Vec<Option<Vec<Float>>> {
        ts.iter().map(|&t| self.evaluate(t)).collect()
    }

    fn find_segment(&self, t: Float) -> Option<&Segment> {
        for seg in &self.segs {
            let left = seg.xold.min(seg.xold + seg.h);
            let right = seg.xold.max(seg.xold + seg.h);
            if t >= left && t <= right {
                return Some(seg);
            }
        }
        None
    }
}
