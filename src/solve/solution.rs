//! Rich solution type for solve_ivp: sampled data, stats, and dense evaluation helpers.

use crate::{Float, error::Error, solve::cont::ContinuousOutput, status::Status};

/// Rich solution of solve_ivp: sampled data plus basic stats
#[derive(Debug, Clone)]
pub struct Solution {
    pub t: Vec<Float>,
    pub y: Vec<Vec<Float>>,
    pub nfev: usize,
    pub nstep: usize,
    pub naccpt: usize,
    pub nrejct: usize,
    pub status: Status,
    pub continuous_sol: Option<ContinuousOutput>,
}

impl Solution {
    /// Evaluate the continuous solution at a single time t.
    /// Returns an error if continuous_sol was disabled or t is outside the covered range.
    pub fn sol(&self, t: Float) -> Result<Vec<Float>, Error> {
        let dense = self
            .continuous_sol
            .as_ref()
            .ok_or(Error::DenseOutputDisabled)?;
        let (start, end) = dense.t_span().ok_or(Error::DenseOutputDisabled)?;
        let (lo, hi) = (start.min(end), start.max(end));
        if t < lo || t > hi {
            return Err(Error::EvaluationOutOfRange(t));
        }
        dense.evaluate(t).ok_or(Error::EvaluationOutOfRange(t))
    }

    /// Evaluate the continuous solution at many time points.
    /// Returns an error if dense output is disabled or any point is outside the covered range.
    pub fn sol_many(&self, ts: &[Float]) -> Result<Vec<Vec<Float>>, Error> {
        let dense = self
            .continuous_sol
            .as_ref()
            .ok_or(Error::DenseOutputDisabled)?;
        let (start, end) = dense.t_span().ok_or(Error::DenseOutputDisabled)?;
        let (lo, hi) = (start.min(end), start.max(end));
        for &t in ts {
            if t < lo || t > hi {
                return Err(Error::EvaluationOutOfRange(t));
            }
        }
        let results = dense.evaluate_many(ts);
        Ok(results.into_iter().map(|opt| opt.unwrap()).collect())
    }

    /// Return the time span covered by the dense output if available.
    pub fn sol_span(&self) -> Option<(Float, Float)> {
        self.continuous_sol.as_ref()?.t_span()
    }

    /// Iterate over stored sample pairs (t_i, y_i) from the discrete output.
    pub fn iter(&self) -> SolutionIter<'_> {
        SolutionIter {
            t_iter: self.t.iter(),
            y_iter: self.y.iter(),
        }
    }
}

/// Iterator over (t, y) pairs of stored samples in an Solution.
pub struct SolutionIter<'a> {
    t_iter: std::slice::Iter<'a, Float>,
    y_iter: std::slice::Iter<'a, Vec<Float>>,
}

impl<'a> Iterator for SolutionIter<'a> {
    type Item = (Float, &'a [Float]);

    fn next(&mut self) -> Option<Self::Item> {
        match (self.t_iter.next(), self.y_iter.next()) {
            (Some(&t), Some(y)) => Some((t, y.as_slice())),
            _ => None,
        }
    }
}
