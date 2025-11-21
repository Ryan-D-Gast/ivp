#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::{PyTuple, PyDict, PyList};
#[cfg(feature = "python")]
use numpy::{PyArray1, PyReadonlyArray1, PyArrayMethods};
#[cfg(feature = "python")]
use crate::{ivp::IVP, Float, solve::solve_ivp, solve::Options, solve::Method, solve::event::{EventConfig, Direction}, solve::cont::ContinuousOutput};

#[cfg(feature = "python")]
#[pyclass(name = "OdeSolution")]
struct PyOdeSolution {
    inner: ContinuousOutput,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyOdeSolution {
    fn __repr__(&self) -> String {
        if let Some((t0, tf)) = self.inner.t_span() {
            format!("<OdeSolution: t_min={:.4}, t_max={:.4}>", t0, tf)
        } else {
            "<OdeSolution: empty>".to_string()
        }
    }

    #[getter]
    fn t_min(&self) -> Option<Float> {
        self.inner.t_span().map(|(t0, _)| t0)
    }

    #[getter]
    fn t_max(&self) -> Option<Float> {
        self.inner.t_span().map(|(_, tf)| tf)
    }

    fn __call__<'py>(&self, py: Python<'py>, t: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        // t can be float or array
        if let Ok(t_val) = t.extract::<Float>() {
            if let Some(y) = self.inner.evaluate(t_val) {
                return Ok(PyArray1::from_vec(py, y).into_any());
            } else {
                return Err(pyo3::exceptions::PyValueError::new_err("t is outside the solution range"));
            }
        } else if let Ok(t_arr) = t.extract::<PyReadonlyArray1<Float>>() {
            let t_slice = t_arr.as_slice()?;
            
            if t_slice.is_empty() {
                 return Ok(PyArray1::from_vec(py, Vec::<Float>::new()).into_any());
            }
            
            let mut flat_results = Vec::new();
            let mut n_states = 0;
            
            for (i, &ti) in t_slice.iter().enumerate() {
                if let Some(yi) = self.inner.evaluate(ti) {
                    if i == 0 {
                        n_states = yi.len();
                        flat_results.reserve(t_slice.len() * n_states);
                    }
                    flat_results.extend(yi);
                } else {
                     return Err(pyo3::exceptions::PyValueError::new_err(format!("t={} is outside the solution range", ti)));
                }
            }
            
            // Result shape: (n_states, n_points) to match SciPy
            let n_points = t_slice.len();
            let mut transposed = vec![0.0; n_points * n_states];
            for i in 0..n_points {
                for j in 0..n_states {
                    transposed[j * n_points + i] = flat_results[i * n_states + j];
                }
            }
            
            let arr = PyArray1::from_vec(py, transposed).reshape((n_states, n_points))?;
            return Ok(arr.into_any());
        }
        
        Err(pyo3::exceptions::PyTypeError::new_err("t must be float or 1D array"))
    }
}

#[cfg(feature = "python")]
#[pyclass(name = "OdeResult", module = "ivp")]
struct PyOdeResult {
    #[pyo3(get)]
    t: Py<PyAny>,
    #[pyo3(get)]
    y: Py<PyAny>,
    #[pyo3(get)]
    t_events: Option<Py<PyAny>>,
    #[pyo3(get)]
    y_events: Option<Py<PyAny>>,
    #[pyo3(get)]
    nfev: usize,
    #[pyo3(get)]
    njev: usize,
    #[pyo3(get)]
    nlu: usize,
    #[pyo3(get)]
    status: i32,
    #[pyo3(get)]
    message: String,
    #[pyo3(get)]
    success: bool,
    #[pyo3(get)]
    sol: Option<Py<PyOdeSolution>>,
}

#[cfg(feature = "python")]
#[pymethods]
impl PyOdeResult {
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
            "success" => Ok(pyo3::types::PyBool::new(py, self.success).as_any().clone().unbind()),
            "sol" => match &self.sol {
                Some(v) => Ok(v.bind(py).clone().into_any().unbind()),
                None => Ok(py.None()),
            },
            _ => Err(pyo3::exceptions::PyKeyError::new_err(key.to_string())),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "  message: {}\n  success: {}\n   status: {}\n     nfev: {}\n     njev: {}\n      nlu: {}\n        t: [{} points]\n        y: [{} states, {} points]",
            self.message, self.success, self.status, self.nfev, self.njev, self.nlu,
            "?", "?", "?" 
        )
    }
}

#[cfg(feature = "python")]
struct PythonIVP<'py> {
    fun: Bound<'py, PyAny>,
    events: Vec<Bound<'py, PyAny>>,
    args: Option<Bound<'py, PyTuple>>,
    event_configs: Vec<EventConfig>,
    py: Python<'py>,
}

#[cfg(feature = "python")]
impl<'py> IVP for PythonIVP<'py> {
    fn ode(&self, x: Float, y: &[Float], dydx: &mut [Float]) {
        // Convert y to numpy array
        let y_arr = PyArray1::from_slice(self.py, y);
        
        // Prepare args
        let args = if let Some(args) = &self.args {
            // Create a new tuple with (x, y_arr, *args)
            // But wait, scipy signature is fun(t, y, *args)
            // PyO3 call needs a tuple of arguments.
            // If args is present, we need to prepend t and y.
            let mut call_args = Vec::with_capacity(2 + args.len());
            call_args.push(x.into_pyobject(self.py).unwrap().into_any());
            call_args.push(y_arr.into_any());
            for arg in args.iter() {
                call_args.push(arg);
            }
            PyTuple::new(self.py, call_args).unwrap()
        } else {
            PyTuple::new(self.py, &[x.into_pyobject(self.py).unwrap().into_any(), y_arr.into_any()]).unwrap()
        };

        // Call function
        let result = self.fun.call1(args).expect("Failed to call ODE function");
        
        // Parse result
        // Result should be array_like
        if let Ok(res_arr) = result.extract::<PyReadonlyArray1<Float>>() {
            let res_slice = res_arr.as_slice().expect("Failed to get slice from result");
            if res_slice.len() != dydx.len() {
                panic!("Derivative shape mismatch");
            }
            dydx.copy_from_slice(res_slice);
        } else if let Ok(res_list) = result.cast::<PyList>() {
             if res_list.len() != dydx.len() {
                panic!("Derivative shape mismatch");
            }
            for (i, item) in res_list.iter().enumerate() {
                dydx[i] = item.extract::<Float>().expect("Failed to extract float from result list");
            }
        } else {
            panic!("ODE function must return array_like");
        }
    }

    fn events(&self, x: Float, y: &[Float], out: &mut [Float]) {
        let y_arr = PyArray1::from_slice(self.py, y);
        
        for (i, event_fun) in self.events.iter().enumerate() {
             let args = if let Some(args) = &self.args {
                let mut call_args = Vec::with_capacity(2 + args.len());
                call_args.push(x.into_pyobject(self.py).unwrap().into_any());
                call_args.push(y_arr.clone().into_any());
                for arg in args.iter() {
                    call_args.push(arg);
                }
                PyTuple::new(self.py, call_args).unwrap()
            } else {
                PyTuple::new(self.py, &[x.into_pyobject(self.py).unwrap().into_any(), y_arr.clone().into_any()]).unwrap()
            };

            let result = event_fun.call1(args).expect("Failed to call event function");
            out[i] = result.extract::<Float>().expect("Event function must return float");
        }
    }

    fn n_events(&self) -> usize {
        self.events.len()
    }

    fn event_config(&self, index: usize) -> EventConfig {
        self.event_configs[index]
    }
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "solve_ivp")]
#[pyo3(signature = (fun, t_span, y0, method=None, t_eval=None, dense_output=false, events=None, vectorized=false, args=None, **options))]
/// Solve an initial value problem for a system of ODEs.
///
/// This function numerically integrates a system of ordinary differential
/// equations given an initial value::
///
///     dy / dt = f(t, y)
///     y(t0) = y0
///
/// Parameters
/// ----------
/// fun : callable
///     Right-hand side of the system. The calling signature is ``fun(t, y)``.
///     Here ``t`` is a scalar, and there are two options for the ndarray ``y``:
///     It can either have shape (n,) then ``fun`` must return array_like with
///     shape (n,). Alternatively, it can have shape (n, k) then ``fun``
///     must return an array_like with shape (n, k), i.e., each column
///     corresponds to a single column in ``y``. The choice between the two
///     options is determined by ``vectorized`` argument (see below).
///     Additional arguments need to be passed if ``args`` is used (see
///     documentation of ``args`` argument).
/// t_span : 2-tuple of floats
///     Interval of integration (t0, tf). The solver starts with t=t0 and
///     integrates until it reaches t=tf.
/// y0 : array_like, shape (n,)
///     Initial state.
/// method : string, optional
///     Integration method to use:
///         * 'RK45' (default): Explicit Runge-Kutta method of order 5(4).
///         * 'RK23': Explicit Runge-Kutta method of order 3(2).
///         * 'DOP853': Explicit Runge-Kutta method of order 8.
///         * 'Radau': Implicit Runge-Kutta method of the Radau IIA family of order 5.
///         * 'BDF': Implicit multi-step variable-order (1 to 5) method.
///         * 'RK4': Explicit Runge-Kutta method of order 4.
/// t_eval : array_like or None, optional
///     Times at which to store the computed solution, must be sorted and lie
///     within `t_span`. If None (default), use points selected by the solver.
/// dense_output : bool, optional
///     Whether to compute a continuous solution. Default is False.
/// events : callable, or list of callables, optional
///     Events to track. If None (default), no events will be tracked.
/// vectorized : bool, optional
///     Whether `fun` can be called in a vectorized fashion. Default is False.
/// args : tuple, optional
///     Additional arguments to pass to the user-defined functions.
/// **options
///     Options passed to the solver:
///         * rtol, atol : float or array_like, optional
///           Relative and absolute tolerances.
///         * first_step : float, optional
///           Initial step size.
///         * max_step : float, optional
///           Maximum allowed step size.
///
/// Returns
/// -------
/// dict with the following fields defined:
/// t : ndarray, shape (n_points,)
///     Time points.
/// y : ndarray, shape (n, n_points)
///     Values of the solution at `t`.
/// t_events : list of ndarray or None
///     Contains for each event type a list of arrays at which an event of
///     that type event was detected.
/// y_events : list of ndarray or None
///     For each value of `t_events`, the corresponding value of the solution.
/// nfev : int
///     Number of evaluations of the right-hand side.
/// njev : int
///     Number of evaluations of the Jacobian.
/// nlu : int
///     Number of LU decompositions.
/// status : int
///     Reason for algorithm termination:
///         * -1: Integration step failed.
///         *  0: The solver successfully reached the end of `tspan`.
///         *  1: A termination event occurred.
/// message : string
///     Human-readable description of the termination reason.
/// success : bool
///     True if the solver reached the interval end or a termination event
///     occurred (``status >= 0``).
fn solve_ivp_py<'py>(
    py: Python<'py>,
    fun: Bound<'py, PyAny>,
    t_span: (f64, f64),
    y0: Bound<'py, PyAny>,
    method: Option<Bound<'py, PyAny>>,
    t_eval: Option<Bound<'py, PyAny>>,
    dense_output: bool,
    events: Option<Bound<'py, PyAny>>,
    vectorized: bool,
    args: Option<Bound<'py, PyTuple>>,
    options: Option<Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = vectorized;
    
    // Parse y0
    let y0_vec: Vec<Float> = if let Ok(arr) = y0.extract::<PyReadonlyArray1<Float>>() {
        arr.as_slice()?.to_vec()
    } else if let Ok(lst) = y0.cast::<PyList>() {
        lst.iter().map(|x| x.extract::<Float>()).collect::<Result<Vec<_>, _>>()?
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err("y0 must be array_like"));
    };

    // Parse method
    let method_enum = if let Some(m) = method {
        if let Ok(s) = m.extract::<String>() {
            Method::from(s.as_str())
        } else {
            Method::DOPRI5
        }
    } else {
        Method::DOPRI5
    };

    // Parse t_eval
    let t_eval_vec: Option<Vec<Float>> = if let Some(te) = t_eval {
        if let Ok(arr) = te.extract::<PyReadonlyArray1<Float>>() {
            Some(arr.as_slice()?.to_vec())
        } else if let Ok(lst) = te.cast::<PyList>() {
            Some(lst.iter().map(|x| x.extract::<Float>()).collect::<Result<Vec<_>, _>>()?)
        } else {
            None
        }
    } else {
        None
    };

    // Parse events
    let mut event_funs = Vec::new();
    let mut event_configs = Vec::new();

    if let Some(ev) = &events {
        if let Ok(lst) = ev.cast::<PyList>() {
            for item in lst.iter() {
                event_funs.push(item.clone());
            }
        } else {
            // Single callable
            event_funs.push(ev.clone());
        }
    }

    // Extract event attributes (terminal, direction)
    for ev in &event_funs {
        let mut config = EventConfig::new();
        if let Ok(term) = ev.getattr("terminal") {
            if let Ok(is_term) = term.extract::<bool>() {
                if is_term {
                    config.terminal();
                }
            }
        }
        if let Ok(dir) = ev.getattr("direction") {
            if let Ok(d) = dir.extract::<f64>() {
                config.direction(Direction::from(d as i32));
            }
        }
        event_configs.push(config);
    }

    // Extract options
    let mut rtol_opt: Option<Float> = None;
    let mut atol_opt: Option<Float> = None;
    let mut max_step_opt: Option<Float> = None;
    let mut first_step_opt: Option<Float> = None;

    if let Some(opts) = options {
        if let Ok(Some(r)) = opts.get_item("rtol") {
             if let Ok(val) = r.extract::<Float>() {
                 rtol_opt = Some(val);
             }
        }
        if let Ok(Some(a)) = opts.get_item("atol") {
             if let Ok(val) = a.extract::<Float>() {
                 atol_opt = Some(val);
             }
        }
        if let Ok(Some(m)) = opts.get_item("max_step") {
             if let Ok(val) = m.extract::<Float>() {
                 max_step_opt = Some(val);
             }
        }
        if let Ok(Some(f)) = opts.get_item("first_step") {
             if let Ok(val) = f.extract::<Float>() {
                 first_step_opt = Some(val);
             }
        }
    }

    // Build Options
    let rtol = rtol_opt.unwrap_or(1e-3);
    let atol = atol_opt.unwrap_or(1e-6);

    let opts = Options::builder()
        .method(method_enum)
        .dense_output(dense_output)
        .maybe_t_eval(t_eval_vec)
        .maybe_max_step(max_step_opt)
        .maybe_first_step(first_step_opt)
        .rtol(rtol)
        .atol(atol)
        .build();

    let python_ivp = PythonIVP {
        fun,
        events: event_funs,
        args,
        event_configs,
        py,
    };

    let result = solve_ivp(&python_ivp, t_span.0, t_span.1, &y0_vec, opts);

    match result {
        Ok(sol) => {
            // Convert Solution to OdeResult (Bunch)
            // We can return a simple dict or a class. SciPy returns an object that inherits from OptimizeResult (which is a dict subclass).
            
            // y is Vec<Vec<Float>> (time, state). SciPy expects (state, time).
            // We need to transpose.
            let n_steps = sol.y.len();
            let n_states = if n_steps > 0 { sol.y[0].len() } else { 0 };
            let mut y_transposed = vec![0.0; n_steps * n_states];
            for (i, step) in sol.y.iter().enumerate() {
                for (j, val) in step.iter().enumerate() {
                    y_transposed[j * n_steps + i] = *val;
                }
            }
            // Create 2D array (n_states, n_steps)
            let y_arr = PyArray1::from_vec(py, y_transposed).reshape((n_states, n_steps))?;

            // t_events, y_events
            let t_events_list = if events.is_some() {
                Some(PyList::new(py, sol.t_events.iter().map(|te| PyArray1::from_vec(py, te.clone())))?.into_any().unbind())
            } else {
                None
            };
            
            // y_events is Vec<Vec<Vec<Float>>>.
            // SciPy: y_events is list of arrays. Each array is (n_events_i, n_states).
            let y_events_list = if events.is_some() {
                let mut y_events_py = Vec::new();
                for ye in sol.y_events {
                    if ye.is_empty() {
                        y_events_py.push(PyList::empty(py).into_any()); // Or empty array? SciPy uses empty array.
                    } else {
                        let n_ev = ye.len();
                        let n_st = ye[0].len();
                        let mut flat = Vec::with_capacity(n_ev * n_st);
                        for state in ye {
                            flat.extend(state);
                        }
                        let arr = PyArray1::from_vec(py, flat).reshape((n_ev, n_st))?;
                        y_events_py.push(arr.into_any());
                    }
                }
                Some(PyList::new(py, y_events_py)?.into_any().unbind())
            } else {
                None
            };
            
            let status_int = match sol.status {
                crate::status::Status::Success => 0,
                crate::status::Status::UserInterrupt => 1,
                _ => -1,
            };

            let sol_obj = if let Some(cont) = sol.continuous_sol {
                Some(Py::new(py, PyOdeSolution { inner: cont })?)
            } else {
                None
            };

            let result_obj = PyOdeResult {
                t: PyArray1::from_vec(py, sol.t).into_any().unbind(),
                y: y_arr.into_any().unbind(),
                t_events: t_events_list,
                y_events: y_events_list,
                nfev: sol.nfev,
                njev: sol.njev,
                nlu: sol.nlu,
                status: status_int,
                message: format!("{:?}", sol.status),
                success: status_int >= 0,
                sol: sol_obj,
            };

            Ok(Bound::new(py, result_obj)?.into_any())
        }
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Solver failed: {:?}", e))),
    }
}

#[cfg(feature = "python")]
#[pymodule]
fn ivp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_ivp_py, m)?)?;
    
    // Add module docstring
    let doc = "A Python interface to the `ivp` Rust crate for solving initial value problems.\n\n\
               This module provides a `solve_ivp` function that mimics the interface of\n\
               `scipy.integrate.solve_ivp`, allowing users to solve systems of ODEs\n\
               using high-performance Rust solvers.\n\n\
               Supported methods:\n\
               - RK45, RK23, DOP853 (Explicit Runge-Kutta)\n\
               - Radau, BDF (Implicit methods for stiff problems)\n\
               - RK4 (Classic Runge-Kutta)\n\n\
               Features:\n\
               - Dense output (continuous solution)\n\
               - Event detection (terminal and direction)\n\
               - Vectorized evaluation (optional)\n\
               - Argument passing to ODE functions";
    m.setattr("__doc__", doc)?;

    Ok(())
}
