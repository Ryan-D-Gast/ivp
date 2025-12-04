//! Dense output wrapper for Python.
//!
//! Provides the `OdeSolution` class that wraps `ContinuousOutput` and allows
//! callable evaluation at arbitrary times within the solution domain.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::solve::cont::ContinuousOutput;
use crate::Float;

/// Python wrapper for dense output interpolation.
///
/// This class is callable and returns the solution at any time within the
/// integration domain. It matches SciPy's `OdeSolution` interface.
#[pyclass(name = "OdeSolution")]
pub struct PyOdeSolution {
    pub(crate) inner: ContinuousOutput,
}

impl PyOdeSolution {
    /// Create a new PyOdeSolution wrapping a ContinuousOutput.
    pub fn new(inner: ContinuousOutput) -> Self {
        Self { inner }
    }

    /// Evaluate at multiple times and return a 2D array (n_states, n_points).
    fn evaluate_array<'py>(
        &self,
        py: Python<'py>,
        t_slice: &[Float],
    ) -> PyResult<Bound<'py, PyAny>> {
        if t_slice.is_empty() {
            return Ok(PyArray1::from_vec(py, Vec::<Float>::new()).into_any());
        }

        let mut flat_results = Vec::new();
        let mut n_states = 0;

        for (i, &ti) in t_slice.iter().enumerate() {
            if let Some(yi) = self.inner.evaluate_extrapolate(ti) {
                if i == 0 {
                    n_states = yi.len();
                    flat_results.reserve(t_slice.len() * n_states);
                }
                flat_results.extend(yi);
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "t={} is outside the solution range",
                    ti
                )));
            }
        }

        // Handle empty state vector
        if n_states == 0 {
            let arr =
                PyArray1::from_vec(py, Vec::<Float>::new()).reshape((0, t_slice.len()))?;
            return Ok(arr.into_any());
        }

        // Transpose to (n_states, n_points) to match SciPy convention
        let n_points = t_slice.len();
        let mut transposed = vec![0.0; n_points * n_states];
        for i in 0..n_points {
            for j in 0..n_states {
                transposed[j * n_points + i] = flat_results[i * n_states + j];
            }
        }

        let arr = PyArray1::from_vec(py, transposed).reshape((n_states, n_points))?;
        Ok(arr.into_any())
    }
}

#[pymethods]
impl PyOdeSolution {
    fn __repr__(&self) -> String {
        if let Some((t0, tf)) = self.inner.t_span() {
            format!("<OdeSolution: t_min={:.4}, t_max={:.4}>", t0, tf)
        } else {
            "<OdeSolution: empty>".to_string()
        }
    }

    /// Minimum time in the solution domain.
    #[getter]
    fn t_min(&self) -> Option<Float> {
        self.inner.t_span().map(|(t0, _)| t0)
    }

    /// Maximum time in the solution domain.
    #[getter]
    fn t_max(&self) -> Option<Float> {
        self.inner.t_span().map(|(_, tf)| tf)
    }

    /// Evaluate the solution at time(s) t.
    ///
    /// Parameters
    /// ----------
    /// t : float or array_like
    ///     Time(s) at which to evaluate the solution.
    ///
    /// Returns
    /// -------
    /// y : ndarray
    ///     Solution at time(s) t. Shape is (n,) for scalar t, or (n, len(t)) for array t.
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        t: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Single float
        if let Ok(t_val) = t.extract::<Float>() {
            if let Some(y) = self.inner.evaluate_extrapolate(t_val) {
                return Ok(PyArray1::from_vec(py, y).into_any());
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "t is outside the solution range",
                ));
            }
        }

        // Numpy array
        if let Ok(t_arr) = t.extract::<PyReadonlyArray1<Float>>() {
            let t_slice = t_arr.as_slice()?;
            return self.evaluate_array(py, t_slice);
        }

        // Python list
        if let Ok(t_list) = t.extract::<Vec<Float>>() {
            return self.evaluate_array(py, &t_list);
        }

        Err(pyo3::exceptions::PyTypeError::new_err(
            "t must be float or 1D array",
        ))
    }
}
