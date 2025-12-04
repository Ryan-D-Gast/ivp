//! IVP trait implementation for Python callables.
//!
//! Wraps Python ODE functions and event functions so they can be used with
//! the Rust solver infrastructure.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::ivp::IVP;
use crate::matrix::Matrix;
use crate::solve::event::EventConfig;
use crate::Float;

use super::sparsity::{SparsityStructure, sparse_jacobian_fd};

/// Wrapper that implements `IVP` for Python ODE functions.
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
        // Try float64 numpy array (most common)
        if let Ok(res_arr) = result.extract::<PyReadonlyArray1<Float>>() {
            let res_slice = res_arr.as_slice().expect("Failed to obtain contiguous slice from numpy array returned by Python ODE function");
            debug_assert_eq!(res_slice.len(), dydx.len(), "Derivative shape mismatch");
            dydx.copy_from_slice(res_slice);
            return;
        }

        // Handle integer numpy arrays
        if let Ok(res_arr) = result.extract::<PyReadonlyArray1<i64>>() {
            let res_slice = res_arr.as_slice().expect("Failed to obtain contiguous slice from integer numpy array returned by Python ODE function");
            debug_assert_eq!(res_slice.len(), dydx.len(), "Derivative shape mismatch");
            for (i, &val) in res_slice.iter().enumerate() {
                dydx[i] = val as Float;
            }
            return;
        }

        if let Ok(res_arr) = result.extract::<PyReadonlyArray1<i32>>() {
            let res_slice = res_arr.as_slice().expect("Failed to obtain contiguous slice from integer numpy array returned by Python ODE function");
            debug_assert_eq!(res_slice.len(), dydx.len(), "Derivative shape mismatch");
            for (i, &val) in res_slice.iter().enumerate() {
                dydx[i] = val as Float;
            }
            return;
        }

        // Python list
        if let Ok(res_list) = result.cast::<PyList>() {
            debug_assert_eq!(res_list.len(), dydx.len(), "Derivative shape mismatch");
            for (i, item) in res_list.iter().enumerate() {
                dydx[i] = item
                    .extract::<Float>()
                    .expect(&format!("Failed to extract float from result list at index {}", i));
            }
            return;
        }

        // Tuple/sequence as Vec
        if let Ok(res_tuple) = result.extract::<Vec<Float>>() {
            debug_assert_eq!(res_tuple.len(), dydx.len(), "Derivative shape mismatch");
            dydx.copy_from_slice(&res_tuple);
            return;
        }

        panic!("ODE function must return an array-like object (list, tuple, or numpy array)");
    }

    /// Parse a 2D matrix result from Python into our Matrix type.
    fn parse_matrix(&self, result: &Bound<'py, PyAny>, j: &mut Matrix) {
        let dim = j.nrows();
        
        // Try float64 numpy 2D array (most common)
        if let Ok(res_arr) = result.extract::<PyReadonlyArray2<Float>>() {
            let shape = res_arr.shape();
            debug_assert_eq!(shape[0], dim, "Jacobian row dimension mismatch");
            debug_assert_eq!(shape[1], dim, "Jacobian column dimension mismatch");
            
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
            debug_assert_eq!(shape[0], dim, "Jacobian row dimension mismatch");
            debug_assert_eq!(shape[1], dim, "Jacobian column dimension mismatch");
            
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
            debug_assert_eq!(shape[0], dim, "Jacobian row dimension mismatch");
            debug_assert_eq!(shape[1], dim, "Jacobian column dimension mismatch");
            
            for row in 0..dim {
                for col in 0..dim {
                    j[(row, col)] = res_arr.get([row, col]).copied().unwrap_or(0) as Float;
                }
            }
            return;
        }

        // Try scipy sparse matrix - convert to dense via toarray()
        if let Ok(to_array) = result.getattr("toarray") {
            if let Ok(dense) = to_array.call0() {
                // Recursively parse the dense array
                self.parse_matrix(&dense, j);
                return;
            }
        }

        panic!("Jacobian must be a 2D array or sparse matrix (e.g. numpy array or scipy sparse matrix)");
    }

    /// Finite difference Jacobian approximation (default fallback).
    fn jac_fd(&self, x: Float, y: &[Float], j: &mut Matrix) {
        let dim = y.len();
        let mut f_origin = vec![0.0; dim];

        // Compute the unperturbed derivative
        self.ode(x, y, &mut f_origin);

        // Use sparse FD if sparsity structure is known
        if let Some(sparsity) = &self.jac_sparsity {
            // Create a closure that captures self for the ODE call
            let ode_fn = |t: Float, y: &[Float], dydx: &mut [Float]| {
                self.ode(t, y, dydx);
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
            self.ode(x, &y_perturbed, &mut f_perturbed);
            y_perturbed[col] = y_original_j;

            for row in 0..dim {
                j[(row, col)] = (f_perturbed[row] - f_origin[row]) / perturbation;
            }
        }
    }
}

impl<'py> IVP for PythonIVP<'py> {
    #[inline]
    fn ode(&self, x: Float, y: &[Float], dydx: &mut [Float]) {
        let y_arr = PyArray1::from_slice(self.py, y);
        let args = self.build_call_args(x, y_arr);

        let result = match self.fun.call1(args) {
            Ok(r) => r,
            Err(e) => panic!("ODE function raised an exception: {}", e),
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
                    Err(e) => panic!("Jacobian function raised an exception: {}", e),
                };

                self.parse_matrix(&result, j);
            } else {
                // Constant matrix - extract once
                self.parse_matrix(jac_fn, j);
            }
        } else {
            // No Jacobian provided - use finite differences (default implementation)
            // Call the default implementation from IVP trait
            self.jac_fd(x, y, j);
        }
    }

    fn events(&self, x: Float, y: &[Float], out: &mut [Float]) {
        let y_arr = PyArray1::from_slice(self.py, y);

        for (i, event_fun) in self.events.iter().enumerate() {
            let args = self.build_call_args(x, y_arr.clone());

            let result = event_fun
                .call1(args)
                .expect(&format!("Failed to call event function at index {}", i));

            out[i] = result
                .extract::<Float>()
                .expect(&format!("Event function at index {} must return a float", i));
        }
    }

    fn n_events(&self) -> usize {
        self.events.len()
    }

    fn event_config(&self, index: usize) -> EventConfig {
        self.event_configs[index]
    }
}
