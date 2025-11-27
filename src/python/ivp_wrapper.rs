//! IVP trait implementation for Python callables.
//!
//! Wraps Python ODE functions and event functions so they can be used with
//! the Rust solver infrastructure.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};

use crate::ivp::IVP;
use crate::solve::event::EventConfig;
use crate::Float;

/// Wrapper that implements `IVP` for Python ODE functions.
///
/// Handles calling Python functions with the appropriate arguments and
/// converting return values back to Rust arrays.
pub struct PythonIVP<'py> {
    fun: Bound<'py, PyAny>,
    events: Vec<Bound<'py, PyAny>>,
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
    /// * `args` - Additional arguments to pass to `fun` and events
    /// * `event_configs` - Configuration for each event (terminal, direction)
    /// * `py` - Python interpreter handle
    pub fn new(
        fun: Bound<'py, PyAny>,
        events: Vec<Bound<'py, PyAny>>,
        args: Option<Bound<'py, PyTuple>>,
        event_configs: Vec<EventConfig>,
        py: Python<'py>,
    ) -> Self {
        Self {
            fun,
            events,
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
            let res_slice = res_arr.as_slice().expect("Failed to get slice from result");
            debug_assert_eq!(res_slice.len(), dydx.len(), "Derivative shape mismatch");
            dydx.copy_from_slice(res_slice);
            return;
        }

        // Handle integer numpy arrays
        if let Ok(res_arr) = result.extract::<PyReadonlyArray1<i64>>() {
            let res_slice = res_arr.as_slice().expect("Failed to get slice from result");
            debug_assert_eq!(res_slice.len(), dydx.len(), "Derivative shape mismatch");
            for (i, &val) in res_slice.iter().enumerate() {
                dydx[i] = val as Float;
            }
            return;
        }

        if let Ok(res_arr) = result.extract::<PyReadonlyArray1<i32>>() {
            let res_slice = res_arr.as_slice().expect("Failed to get slice from result");
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
                    .expect("Failed to extract float from result list");
            }
            return;
        }

        // Tuple/sequence as Vec
        if let Ok(res_tuple) = result.extract::<Vec<Float>>() {
            debug_assert_eq!(res_tuple.len(), dydx.len(), "Derivative shape mismatch");
            dydx.copy_from_slice(&res_tuple);
            return;
        }

        panic!("ODE function must return array_like");
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

    fn events(&self, x: Float, y: &[Float], out: &mut [Float]) {
        let y_arr = PyArray1::from_slice(self.py, y);

        for (i, event_fun) in self.events.iter().enumerate() {
            let args = self.build_call_args(x, y_arr.clone());

            let result = event_fun
                .call1(args)
                .expect("Failed to call event function");

            out[i] = result
                .extract::<Float>()
                .expect("Event function must return float");
        }
    }

    fn n_events(&self) -> usize {
        self.events.len()
    }

    fn event_config(&self, index: usize) -> EventConfig {
        self.event_configs[index]
    }
}
