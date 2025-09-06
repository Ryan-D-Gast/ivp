//! Rich solution type for solve_ivp: sampled data, stats, and dense evaluation helpers.

use crate::{
    Float,
    core::status::Status,
    solve::cont::ContinuousOutput,
};

/// Rich solution of solve_ivp: sampled data plus basic stats
#[derive(Debug, Clone)]
pub struct IVPSolution {
    pub t: Vec<Float>,
    pub y: Vec<Vec<Float>>,
    pub nfev: usize,
    pub nstep: usize,
    pub naccpt: usize,
    pub nrejct: usize,
    pub status: Status,
    pub(crate) continuous_sol: Option<ContinuousOutput>,
}

impl IVPSolution {
    /// Evaluate the continuous solution at a single time t.
    /// Returns None if continuous_sol was disabled or t is outside the covered range.
    pub fn sol(&self, t: Float) -> Option<Vec<Float>> {
        self.continuous_sol.as_ref()?.evaluate(t)
    }

    /// Evaluate the continuous solution at many time points.
    /// If dense output is disabled, returns a Vec of None of the same length.
    /// Points outside the range yield None entries.
    pub fn sol_many(&self, ts: &[Float]) -> Vec<Option<Vec<Float>>> {
        match self.continuous_sol.as_ref() {
            Some(dense) => dense.evaluate_many(ts),
            None => vec![None; ts.len()],
        }
    }

    /// Return the time span covered by the dense output if available.
    pub fn sol_span(&self) -> Option<(Float, Float)> {
        self.continuous_sol.as_ref()?.t_span()
    }

    /// Iterate over stored sample pairs (t_i, y_i) from the discrete output.
    pub fn iter(&self) -> SolutionIter {
        SolutionIter { t_iter: self.t.iter(), y_iter: self.y.iter() }
    }
}

/// Iterator over (t, y) pairs of stored samples in an IVPSolution.
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
