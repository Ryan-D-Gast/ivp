//! Result object for Python.
//!
//! Provides the `OdeResult` class that contains the solution and metadata,
//! matching SciPy's return value structure.

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use numpy::{PyArray1, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;

use crate::Float;
use crate::solve::adjoint::AdjointSolver;
use super::solution::PyOdeSolution;
use super::ivp_wrapper::PythonIVP;
use super::conversion::extract_float_array;

/// Result object returned by `solve_ivp`.
///
/// This class mimics SciPy's `OdeResult` (a Bunch subclass), providing both
/// attribute and dictionary-style access to solution data.
#[pyclass(name = "OdeResult", module = "ivp")]
pub struct PyOdeResult {
    /// Original ODE function.
    #[pyo3(get)]
    pub fun: Py<PyAny>,

    /// Additional arguments for the ODE function.
    #[pyo3(get)]
    pub args: Option<Py<PyTuple>>,

    /// Parameters for the ODE system.
    #[pyo3(get)]
    pub p: Option<Py<PyAny>>,

    /// Time points of the solution.
    #[pyo3(get)]
    pub t: Py<PyAny>,

    /// Solution values at each time point. Shape: (n_states, n_points).
    #[pyo3(get)]
    pub y: Py<PyAny>,

    /// Sensitivity values dy/dp at each time point. Shape: (n_params, n_states, n_points).
    #[pyo3(get)]
    pub s: Option<Py<PyAny>>,

    /// Times at which events occurred. List of arrays, one per event function.
    #[pyo3(get)]
    pub t_events: Option<Py<PyAny>>,

    /// Solution values at event times. List of arrays, one per event function.
    #[pyo3(get)]
    pub y_events: Option<Py<PyAny>>,

    /// Numerical quadrature results.
    #[pyo3(get)]
    pub quad: Option<Py<PyAny>>,

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
    /// Compute parameter gradients using adjoint sensitivity analysis.
    ///
    /// Requires that the forward pass was performed with `dense_output=True`.
    ///
    /// Parameters
    /// ----------
    /// lambda_tf : array_like, shape (n_states,)
    ///     Terminal condition for the adjoint state (d cost / dy at tf).
    /// dgdy : callable
    ///     Gradient of the running cost g(t, y, p) with respect to y: `dgdy(t, y, p) -> array_like`.
    /// dgdp : callable
    ///     Gradient of the running cost g(t, y, p) with respect to p: `dgdp(t, y, p) -> array_like`.
    /// dhdp : callable
    ///     Gradient of the terminal cost h(y_tf, p) with respect to p: `dhdp(y_tf, p) -> array_like`.
    ///
    /// Returns
    /// -------
    /// gradient : ndarray, shape (n_params,)
    ///     Gradient of the total cost with respect to parameters p.
    fn adjoint_solve<'py>(
        &self,
        py: Python<'py>,
        lambda_tf: Bound<'py, PyAny>,
        dgdy: Bound<'py, PyAny>,
        dgdp: Bound<'py, PyAny>,
        dhdp: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let py_sol = self.sol.as_ref().ok_or_else(|| {
            PyValueError::new_err("Adjoint solve requires dense_output=True in the forward pass")
        })?;
        let sol_bind = py_sol.bind(py);
        let sol_inner = &sol_bind.borrow().inner;

        // Reconstruct a minimal Rust Solution object from ContinuousOutput
        let forward_sol = crate::solve::Solution {
            t: Vec::new(),
            y: Vec::new(),
            t_events: Vec::new(),
            y_events: Vec::new(),
            quad: Vec::new(),
            nfev: 0,
            njev: 0,
            nlu: 0,
            nstep: 0,
            naccpt: 0,
            nrejct: 0,
            status: crate::status::Status::Success,
            continuous_sol: Some(sol_inner.clone()),
        };

        let p_vec = match &self.p {
            Some(pv) => extract_float_array(pv.bind(py))?,
            None => Vec::new(),
        };

        // Reconstruct PythonIVP wrapper
        let python_ivp = PythonIVP::new(
            self.fun.bind(py).clone(),
            Vec::new(),
            None,
            None,
            None,
            self.args.as_ref().map(|a| a.bind(py).clone()),
            Vec::new(),
            py,
        );

        let adjoint_solver = AdjointSolver::new(&python_ivp, &forward_sol);

        let lambda_tf_vec = extract_float_array(&lambda_tf)?;

        // Callbacks for adjoint solver
        let dgdy_cb = |t: Float, y: &[Float], p: &[Float], out: &mut [Float]| {
            let y_arr = PyArray1::from_slice(py, y);
            let p_arr = PyArray1::from_slice(py, p);
            let args = (t, y_arr, p_arr);
            let result = dgdy.call1(args).expect("dgdy callback failed");
            extract_vector_to_slice(&result, out);
        };

        let dgdp_cb = |t: Float, y: &[Float], p: &[Float], out: &mut [Float]| {
            let y_arr = PyArray1::from_slice(py, y);
            let p_arr = PyArray1::from_slice(py, p);
            let args = (t, y_arr, p_arr);
            let result = dgdp.call1(args).expect("dgdp callback failed");
            extract_vector_to_slice(&result, out);
        };

        let dhdp_cb = |y_tf: &[Float], p: &[Float], out: &mut [Float]| {
            let y_arr = PyArray1::from_slice(py, y_tf);
            let p_arr = PyArray1::from_slice(py, p);
            let args = (y_arr, p_arr);
            let result = dhdp.call1(args).expect("dhdp callback failed");
            extract_vector_to_slice(&result, out);
        };

        let gradient = adjoint_solver
            .compute_gradient(&p_vec, &lambda_tf_vec, dgdy_cb, dgdp_cb, dhdp_cb)
            .map_err(|e| PyValueError::new_err(format!("Adjoint solve failed: {:?}", e)))?;

        Ok(PyArray1::from_vec(py, gradient).into_any())
    }

    /// Dictionary-style access to result fields.
    fn __getitem__(&self, key: &str, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match key {
            "fun" => Ok(self.fun.clone_ref(py)),
            "args" => match &self.args {
                Some(v) => Ok(v.clone_ref(py).into_any()),
                None => Ok(py.None()),
            },
            "p" => match &self.p {
                Some(v) => Ok(v.clone_ref(py)),
                None => Ok(py.None()),
            },
            "t" => Ok(self.t.clone_ref(py)),
            "y" => Ok(self.y.clone_ref(py)),
            "s" => match &self.s {
                Some(v) => Ok(v.clone_ref(py)),
                None => Ok(py.None()),
            },
            "t_events" => match &self.t_events {
                Some(v) => Ok(v.clone_ref(py)),
                None => Ok(py.None()),
            },
            "y_events" => match &self.y_events {
                Some(v) => Ok(v.clone_ref(py)),
                None => Ok(py.None()),
            },
            "quad" => match &self.quad {
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

fn extract_vector_to_slice(result: &Bound<'_, PyAny>, out: &mut [Float]) {
    let vec = extract_float_array(result).expect("Expected array-like from callback");
    if vec.len() == out.len() {
        out.copy_from_slice(&vec);
    } else {
        panic!("Callback returned vector of wrong size: expected {}, got {}", out.len(), vec.len());
    }
}
