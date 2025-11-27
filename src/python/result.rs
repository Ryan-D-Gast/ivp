//! Result object for Python.
//!
//! Provides the `OdeResult` class that contains the solution and metadata,
//! matching SciPy's return value structure.

use pyo3::prelude::*;

use super::solution::PyOdeSolution;

/// Result object returned by `solve_ivp`.
///
/// This class mimics SciPy's `OdeResult` (a Bunch subclass), providing both
/// attribute and dictionary-style access to solution data.
#[pyclass(name = "OdeResult", module = "ivp")]
pub struct PyOdeResult {
    /// Time points of the solution.
    #[pyo3(get)]
    pub t: Py<PyAny>,

    /// Solution values at each time point. Shape: (n_states, n_points).
    #[pyo3(get)]
    pub y: Py<PyAny>,

    /// Times at which events occurred. List of arrays, one per event function.
    #[pyo3(get)]
    pub t_events: Option<Py<PyAny>>,

    /// Solution values at event times. List of arrays, one per event function.
    #[pyo3(get)]
    pub y_events: Option<Py<PyAny>>,

    /// Number of function evaluations.
    #[pyo3(get)]
    pub nfev: usize,

    /// Number of Jacobian evaluations.
    #[pyo3(get)]
    pub njev: usize,

    /// Number of LU decompositions.
    #[pyo3(get)]
    pub nlu: usize,

    /// Status code: 0 = success, 1 = terminated by event, -1 = failed.
    #[pyo3(get)]
    pub status: i32,

    /// Human-readable termination message.
    #[pyo3(get)]
    pub message: String,

    /// True if integration was successful.
    #[pyo3(get)]
    pub success: bool,

    /// Dense output object for interpolation (if requested).
    #[pyo3(get)]
    pub sol: Option<Py<PyOdeSolution>>,
}

#[pymethods]
impl PyOdeResult {
    /// Dictionary-style access to result fields.
    fn __getitem__(&self, key: &str, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match key {
            "t" => Ok(self.t.clone_ref(py)),
            "y" => Ok(self.y.clone_ref(py)),
            "t_events" => match &self.t_events {
                Some(v) => Ok(v.clone_ref(py)),
                None => Ok(py.None()),
            },
            "y_events" => match &self.y_events {
                Some(v) => Ok(v.clone_ref(py)),
                None => Ok(py.None()),
            },
            "nfev" => Ok(self.nfev.into_pyobject(py)?.into_any().unbind()),
            "njev" => Ok(self.njev.into_pyobject(py)?.into_any().unbind()),
            "nlu" => Ok(self.nlu.into_pyobject(py)?.into_any().unbind()),
            "status" => Ok(self.status.into_pyobject(py)?.into_any().unbind()),
            "message" => Ok(self.message.clone().into_pyobject(py)?.into_any().unbind()),
            "success" => Ok(pyo3::types::PyBool::new(py, self.success)
                .as_any()
                .clone()
                .unbind()),
            "sol" => match &self.sol {
                Some(v) => Ok(v.bind(py).clone().into_any().unbind()),
                None => Ok(py.None()),
            },
            _ => Err(pyo3::exceptions::PyKeyError::new_err(key.to_string())),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "  message: {}\n  success: {}\n   status: {}\n     nfev: {}\n     njev: {}\n      nlu: {}",
            self.message, self.success, self.status, self.nfev, self.njev, self.nlu
        )
    }
}
