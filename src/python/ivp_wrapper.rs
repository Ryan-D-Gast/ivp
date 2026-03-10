//! System trait implementations for Python callables.
//!
//! Wraps Python ODE functions and event functions so they can be used with
//! the Rust solver infrastructure.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use std::panic::panic_any;

use crate::Float;
use crate::ivp::{FirstOrderSystem, SecondOrderSystem, SeparableHamiltonianSystem};
use crate::matrix::Matrix;
use crate::solve::event::EventConfig;

use super::sparsity::{SparsityStructure, sparse_jacobian_fd};

/// Wrapper that implements [`FirstOrderSystem`] for Python ODE functions.
///
/// Handles calling Python functions with the appropriate arguments and
/// converting return values back to Rust arrays.
pub struct PythonIVP<'py> {
    fun: Bound<'py, PyAny>,
    events: Vec<Bound<'py, PyAny>>,
    jac: Option<Bound<'py, PyAny>>,
    jac_sparsity: Option<SparsityStructure>,
    args: Option<Bound<'py, PyTuple>>,
    event_configs: Vec<EventConfig>,
    py: Python<'py>,
}

#[derive(Debug)]
pub enum PythonCallbackErrorKind {
    Type,
    Value,
    Runtime,
}

#[derive(Debug)]
pub struct PythonCallbackError {
    pub kind: PythonCallbackErrorKind,
    pub message: String,
}

fn raise_python_callback_error(kind: PythonCallbackErrorKind, message: impl Into<String>) -> ! {
    panic_any(PythonCallbackError {
        kind,
        message: message.into(),
    });
}

fn build_call_args<'py>(
    py: Python<'py>,
    args: Option<&Bound<'py, PyTuple>>,
    x: Float,
    y_arr: Bound<'py, PyArray1<Float>>,
) -> Bound<'py, PyTuple> {
    if let Some(extra_args) = args {
        let mut call_args = Vec::with_capacity(2 + extra_args.len());
        call_args.push(x.into_pyobject(py).unwrap().into_any());
        call_args.push(y_arr.into_any());
        for arg in extra_args.iter() {
            call_args.push(arg);
        }
        PyTuple::new(py, call_args).unwrap()
    } else {
        PyTuple::new(
            py,
            &[x.into_pyobject(py).unwrap().into_any(), y_arr.into_any()],
        )
        .unwrap()
    }
}

fn parse_vector_result(result: &Bound<'_, PyAny>, out: &mut [Float]) {
    if let Ok(res_arr) = result.extract::<PyReadonlyArray1<Float>>() {
        let res_slice = res_arr.as_slice().unwrap_or_else(|_| {
            raise_python_callback_error(
                PythonCallbackErrorKind::Type,
                "Python callback must return a contiguous 1D NumPy array or other 1D array-like object",
            )
        });
        if res_slice.len() != out.len() {
            raise_python_callback_error(
                PythonCallbackErrorKind::Value,
                format!(
                    "Python callback returned {} values, but {} were expected",
                    res_slice.len(),
                    out.len()
                ),
            );
        }
        out.copy_from_slice(res_slice);
        return;
    }

    if let Ok(res_arr) = result.extract::<PyReadonlyArray2<Float>>() {
        let shape = res_arr.shape();
        if shape[0] == out.len() && shape[1] == 1 {
            for (i, value) in out.iter_mut().enumerate() {
                *value = res_arr.get([i, 0]).copied().unwrap_or(0.0);
            }
            return;
        }
        if shape[0] == 1 && shape[1] == out.len() {
            for (i, value) in out.iter_mut().enumerate() {
                *value = res_arr.get([0, i]).copied().unwrap_or(0.0);
            }
            return;
        }
        raise_python_callback_error(
            PythonCallbackErrorKind::Value,
            format!(
                "Python callback returned a 2D array with shape {:?}; expected ({}, 1) or (1, {})",
                shape,
                out.len(),
                out.len()
            ),
        );
    }

    if let Ok(res_arr) = result.extract::<PyReadonlyArray1<i64>>() {
        let res_slice = res_arr.as_slice().unwrap_or_else(|_| {
            raise_python_callback_error(
                PythonCallbackErrorKind::Type,
                "Python callback must return a contiguous 1D NumPy array or other 1D array-like object",
            )
        });
        if res_slice.len() != out.len() {
            raise_python_callback_error(
                PythonCallbackErrorKind::Value,
                format!(
                    "Python callback returned {} values, but {} were expected",
                    res_slice.len(),
                    out.len()
                ),
            );
        }
        for (i, &val) in res_slice.iter().enumerate() {
            out[i] = val as Float;
        }
        return;
    }

    if let Ok(res_arr) = result.extract::<PyReadonlyArray2<i64>>() {
        let shape = res_arr.shape();
        if shape[0] == out.len() && shape[1] == 1 {
            for (i, value) in out.iter_mut().enumerate() {
                *value = res_arr.get([i, 0]).copied().unwrap_or(0) as Float;
            }
            return;
        }
        if shape[0] == 1 && shape[1] == out.len() {
            for (i, value) in out.iter_mut().enumerate() {
                *value = res_arr.get([0, i]).copied().unwrap_or(0) as Float;
            }
            return;
        }
        raise_python_callback_error(
            PythonCallbackErrorKind::Value,
            format!(
                "Python callback returned a 2D array with shape {:?}; expected ({}, 1) or (1, {})",
                shape,
                out.len(),
                out.len()
            ),
        );
    }

    if let Ok(res_arr) = result.extract::<PyReadonlyArray1<i32>>() {
        let res_slice = res_arr.as_slice().unwrap_or_else(|_| {
            raise_python_callback_error(
                PythonCallbackErrorKind::Type,
                "Python callback must return a contiguous 1D NumPy array or other 1D array-like object",
            )
        });
        if res_slice.len() != out.len() {
            raise_python_callback_error(
                PythonCallbackErrorKind::Value,
                format!(
                    "Python callback returned {} values, but {} were expected",
                    res_slice.len(),
                    out.len()
                ),
            );
        }
        for (i, &val) in res_slice.iter().enumerate() {
            out[i] = val as Float;
        }
        return;
    }

    if let Ok(res_arr) = result.extract::<PyReadonlyArray2<i32>>() {
        let shape = res_arr.shape();
        if shape[0] == out.len() && shape[1] == 1 {
            for (i, value) in out.iter_mut().enumerate() {
                *value = res_arr.get([i, 0]).copied().unwrap_or(0) as Float;
            }
            return;
        }
        if shape[0] == 1 && shape[1] == out.len() {
            for (i, value) in out.iter_mut().enumerate() {
                *value = res_arr.get([0, i]).copied().unwrap_or(0) as Float;
            }
            return;
        }
        raise_python_callback_error(
            PythonCallbackErrorKind::Value,
            format!(
                "Python callback returned a 2D array with shape {:?}; expected ({}, 1) or (1, {})",
                shape,
                out.len(),
                out.len()
            ),
        );
    }

    if let Ok(res_list) = result.cast::<PyList>() {
        if res_list.len() != out.len() {
            raise_python_callback_error(
                PythonCallbackErrorKind::Value,
                format!(
                    "Python callback returned {} values, but {} were expected",
                    res_list.len(),
                    out.len()
                ),
            );
        }
        for (i, item) in res_list.iter().enumerate() {
            out[i] = item.extract::<Float>().unwrap_or_else(|_| {
                raise_python_callback_error(
                    PythonCallbackErrorKind::Type,
                    format!(
                        "Python callback returned a non-numeric value at index {}",
                        i
                    ),
                )
            });
        }
        return;
    }

    if let Ok(res_tuple) = result.extract::<Vec<Float>>() {
        if res_tuple.len() != out.len() {
            raise_python_callback_error(
                PythonCallbackErrorKind::Value,
                format!(
                    "Python callback returned {} values, but {} were expected",
                    res_tuple.len(),
                    out.len()
                ),
            );
        }
        out.copy_from_slice(&res_tuple);
        return;
    }

    raise_python_callback_error(
        PythonCallbackErrorKind::Type,
        "Python callback must return a 1D array-like object",
    );
}

/// Python wrapper for second-order symplectic problems.
pub struct PythonSecondOrderIVP<'py> {
    acceleration: Bound<'py, PyAny>,
    args: Option<Bound<'py, PyTuple>>,
    py: Python<'py>,
}

impl<'py> PythonSecondOrderIVP<'py> {
    pub fn new(
        acceleration: Bound<'py, PyAny>,
        args: Option<Bound<'py, PyTuple>>,
        py: Python<'py>,
    ) -> Self {
        Self {
            acceleration,
            args,
            py,
        }
    }
}

impl SecondOrderSystem for PythonSecondOrderIVP<'_> {
    fn acceleration(&self, t: Float, q: &[Float], a: &mut [Float]) {
        let q_arr = PyArray1::from_slice(self.py, q);
        let args = build_call_args(self.py, self.args.as_ref(), t, q_arr);
        let result = self.acceleration.call1(args).unwrap_or_else(|e| {
            raise_python_callback_error(
                PythonCallbackErrorKind::Runtime,
                format!("acceleration callback raised an exception: {}", e),
            )
        });
        parse_vector_result(&result, a);
    }
}

/// Python wrapper for separable Hamiltonian symplectic problems.
pub struct PythonHamiltonianIVP<'py> {
    position_derivative: Bound<'py, PyAny>,
    momentum_derivative: Bound<'py, PyAny>,
    args: Option<Bound<'py, PyTuple>>,
    py: Python<'py>,
}

impl<'py> PythonHamiltonianIVP<'py> {
    pub fn new(
        position_derivative: Bound<'py, PyAny>,
        momentum_derivative: Bound<'py, PyAny>,
        args: Option<Bound<'py, PyTuple>>,
        py: Python<'py>,
    ) -> Self {
        Self {
            position_derivative,
            momentum_derivative,
            args,
            py,
        }
    }
}

impl SeparableHamiltonianSystem for PythonHamiltonianIVP<'_> {
    fn position_derivative(&self, t: Float, p: &[Float], dqdt: &mut [Float]) {
        let p_arr = PyArray1::from_slice(self.py, p);
        let args = build_call_args(self.py, self.args.as_ref(), t, p_arr);
        let result = self.position_derivative.call1(args).unwrap_or_else(|e| {
            raise_python_callback_error(
                PythonCallbackErrorKind::Runtime,
                format!("position_derivative callback raised an exception: {}", e),
            )
        });
        parse_vector_result(&result, dqdt);
    }

    fn momentum_derivative(&self, t: Float, q: &[Float], dpdt: &mut [Float]) {
        let q_arr = PyArray1::from_slice(self.py, q);
        let args = build_call_args(self.py, self.args.as_ref(), t, q_arr);
        let result = self.momentum_derivative.call1(args).unwrap_or_else(|e| {
            raise_python_callback_error(
                PythonCallbackErrorKind::Runtime,
                format!("momentum_derivative callback raised an exception: {}", e),
            )
        });
        parse_vector_result(&result, dpdt);
    }
}

impl<'py> PythonIVP<'py> {
    /// Create a new PythonIVP wrapper.
    ///
    /// # Arguments
    /// * `fun` - The ODE function `f(t, y, *args)`
    /// * `events` - List of event functions
    /// * `jac` - Optional Jacobian function or constant matrix
    /// * `jac_sparsity` - Optional Jacobian sparsity structure
    /// * `args` - Additional arguments to pass to `fun` and events
    /// * `event_configs` - Configuration for each event (terminal, direction)
    /// * `py` - Python interpreter handle
    pub fn new(
        fun: Bound<'py, PyAny>,
        events: Vec<Bound<'py, PyAny>>,
        jac: Option<Bound<'py, PyAny>>,
        jac_sparsity: Option<SparsityStructure>,
        args: Option<Bound<'py, PyTuple>>,
        event_configs: Vec<EventConfig>,
        py: Python<'py>,
    ) -> Self {
        Self {
            fun,
            events,
            jac,
            jac_sparsity,
            args,
            event_configs,
            py,
        }
    }

    /// Build call arguments tuple: (t, y, *args)
    fn build_call_args(&self, x: Float, y_arr: Bound<'py, PyArray1<Float>>) -> Bound<'py, PyTuple> {
        if let Some(extra_args) = &self.args {
            let mut call_args = Vec::with_capacity(2 + extra_args.len());
            call_args.push(x.into_pyobject(self.py).unwrap().into_any());
            call_args.push(y_arr.into_any());
            for arg in extra_args.iter() {
                call_args.push(arg);
            }
            PyTuple::new(self.py, call_args).unwrap()
        } else {
            PyTuple::new(
                self.py,
                &[
                    x.into_pyobject(self.py).unwrap().into_any(),
                    y_arr.into_any(),
                ],
            )
            .unwrap()
        }
    }

    /// Parse ODE function result into the derivative array.
    fn parse_result(&self, result: &Bound<'py, PyAny>, dydx: &mut [Float]) {
        parse_vector_result(result, dydx);
    }

    /// Parse a 2D matrix result from Python into our Matrix type.
    fn parse_matrix_result(result: &Bound<'py, PyAny>, j: &mut Matrix) {
        let dim = j.nrows();

        // Try float64 numpy 2D array (most common)
        if let Ok(res_arr) = result.extract::<PyReadonlyArray2<Float>>() {
            let shape = res_arr.shape();
            if shape[0] != dim || shape[1] != dim {
                raise_python_callback_error(
                    PythonCallbackErrorKind::Value,
                    format!(
                        "Jacobian must have shape ({0}, {0}), got ({1}, {2})",
                        dim, shape[0], shape[1]
                    ),
                );
            }

            // Copy values row by row
            for row in 0..dim {
                for col in 0..dim {
                    j[(row, col)] = res_arr.get([row, col]).copied().unwrap_or(0.0);
                }
            }
            return;
        }

        // Try int64 numpy 2D array
        if let Ok(res_arr) = result.extract::<PyReadonlyArray2<i64>>() {
            let shape = res_arr.shape();
            if shape[0] != dim || shape[1] != dim {
                raise_python_callback_error(
                    PythonCallbackErrorKind::Value,
                    format!(
                        "Jacobian must have shape ({0}, {0}), got ({1}, {2})",
                        dim, shape[0], shape[1]
                    ),
                );
            }

            for row in 0..dim {
                for col in 0..dim {
                    j[(row, col)] = res_arr.get([row, col]).copied().unwrap_or(0) as Float;
                }
            }
            return;
        }

        // Try int32 numpy 2D array
        if let Ok(res_arr) = result.extract::<PyReadonlyArray2<i32>>() {
            let shape = res_arr.shape();
            if shape[0] != dim || shape[1] != dim {
                raise_python_callback_error(
                    PythonCallbackErrorKind::Value,
                    format!(
                        "Jacobian must have shape ({0}, {0}), got ({1}, {2})",
                        dim, shape[0], shape[1]
                    ),
                );
            }

            for row in 0..dim {
                for col in 0..dim {
                    j[(row, col)] = res_arr.get([row, col]).copied().unwrap_or(0) as Float;
                }
            }
            return;
        }

        // Try scipy sparse matrix - convert to dense via toarray()
        if let Ok(to_array) = result.getattr("toarray")
            && let Ok(dense) = to_array.call0()
        {
            // Recursively parse the dense array
            Self::parse_matrix_result(&dense, j);
            return;
        }

        raise_python_callback_error(
            PythonCallbackErrorKind::Type,
            "Jacobian must be a 2D array or sparse matrix (for example, a NumPy array or SciPy sparse matrix)",
        );
    }

    /// Finite difference Jacobian approximation (default fallback).
    fn jac_fd(&self, x: Float, y: &[Float], j: &mut Matrix) {
        let dim = y.len();
        let mut f_origin = vec![0.0; dim];

        // Compute the unperturbed derivative
        self.derivative(x, y, &mut f_origin);

        // Use sparse FD if sparsity structure is known
        if let Some(sparsity) = &self.jac_sparsity {
            // Create a closure that captures self for the ODE call
            let ode_fn = |t: Float, y: &[Float], dydx: &mut [Float]| {
                self.derivative(t, y, dydx);
            };
            sparse_jacobian_fd(ode_fn, x, y, &f_origin, sparsity, j);
            return;
        }

        // Dense finite differences
        let eps = Float::EPSILON.sqrt();
        let mut y_perturbed = y.to_vec();
        let mut f_perturbed = vec![0.0; dim];

        // For each column of the jacobian
        for col in 0..dim {
            let y_original_j = y[col];
            let perturbation = eps * y_original_j.abs().max(1.0);
            y_perturbed[col] = y_original_j + perturbation;
            self.derivative(x, &y_perturbed, &mut f_perturbed);
            y_perturbed[col] = y_original_j;

            for row in 0..dim {
                j[(row, col)] = (f_perturbed[row] - f_origin[row]) / perturbation;
            }
        }
    }
}

impl<'py> FirstOrderSystem for PythonIVP<'py> {
    #[inline]
    fn derivative(&self, x: Float, y: &[Float], dydx: &mut [Float]) {
        let y_arr = PyArray1::from_slice(self.py, y);
        let args = self.build_call_args(x, y_arr);

        let result = match self.fun.call1(args) {
            Ok(r) => r,
            Err(e) => raise_python_callback_error(
                PythonCallbackErrorKind::Runtime,
                format!("ODE function raised an exception: {}", e),
            ),
        };

        self.parse_result(&result, dydx);
    }

    fn jac(&self, x: Float, y: &[Float], j: &mut Matrix) {
        if let Some(jac_fn) = &self.jac {
            // Check if jac is callable or a constant matrix
            if jac_fn.is_callable() {
                // Call the Jacobian function
                let y_arr = PyArray1::from_slice(self.py, y);
                let args = self.build_call_args(x, y_arr);

                let result = match jac_fn.call1(args) {
                    Ok(r) => r,
                    Err(e) => raise_python_callback_error(
                        PythonCallbackErrorKind::Runtime,
                        format!("Jacobian function raised an exception: {}", e),
                    ),
                };

                Self::parse_matrix_result(&result, j);
            } else {
                // Constant matrix - extract once
                Self::parse_matrix_result(jac_fn, j);
            }
        } else {
            // No Jacobian provided - use finite differences (default implementation)
            // Call the default implementation from FirstOrderSystem trait
            self.jac_fd(x, y, j);
        }
    }

    fn events(&self, x: Float, y: &[Float], out: &mut [Float]) {
        let y_arr = PyArray1::from_slice(self.py, y);

        for (i, event_fun) in self.events.iter().enumerate() {
            let args = self.build_call_args(x, y_arr.clone());

            let result = event_fun.call1(args).unwrap_or_else(|e| {
                raise_python_callback_error(
                    PythonCallbackErrorKind::Runtime,
                    format!("event function at index {} raised an exception: {}", i, e),
                )
            });

            out[i] = result.extract::<Float>().unwrap_or_else(|_| {
                raise_python_callback_error(
                    PythonCallbackErrorKind::Type,
                    format!("event function at index {} must return a float", i),
                )
            });
        }
    }

    fn n_events(&self) -> usize {
        self.events.len()
    }

    fn event_config(&self, index: usize) -> EventConfig {
        self.event_configs[index]
    }
}
