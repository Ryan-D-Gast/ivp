//! Main solve_ivp function for Python.
//!
//! This module contains the `solve_ivp` function that serves as the primary
//! entry point for solving ODEs from Python.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::methods::Tolerance;
use crate::solve::event::{Direction, EventConfig};
use crate::solve::{solve_ivp, Method, Options};
use crate::Float;

use super::conversion::{extract_float_array, parse_t_span};
use super::ivp_wrapper::PythonIVP;
use super::result::PyOdeResult;
use super::solution::PyOdeSolution;
use super::sparsity::SparsityStructure;

/// Solve an initial value problem for a system of ODEs.
///
/// This function numerically integrates a system of ordinary differential
/// equations given an initial value:
///
///     dy / dt = f(t, y)
///     y(t0) = y0
///
/// # Parameters
///
/// * `fun` - Right-hand side of the system. Signature: `fun(t, y) -> array_like`
/// * `t_span` - Integration interval `(t0, tf)`
/// * `y0` - Initial state vector
/// * `method` - Integration method: "RK45", "RK23", "DOP853", "Radau", "BDF", "RK4"
/// * `t_eval` - Times at which to store the solution
/// * `dense_output` - Whether to compute continuous solution
/// * `events` - Event functions to track
/// * `vectorized` - Whether `fun` can be called in vectorized fashion (ignored)
/// * `args` - Additional arguments to pass to `fun` and event functions
/// * `**options` - Additional options: `rtol`, `atol`, `first_step`, `max_step`
///
/// # Returns
///
/// An `OdeResult` object containing:
/// * `t` - Time points
/// * `y` - Solution values (shape: n_states × n_points)
/// * `t_events`, `y_events` - Event times and states
/// * `sol` - Dense output object (if requested)
/// * `nfev`, `njev`, `nlu` - Evaluation counts
/// * `status`, `message`, `success` - Termination info
#[pyfunction]
#[pyo3(name = "solve_ivp")]
#[pyo3(signature = (fun, t_span, y0, method=None, t_eval=None, dense_output=false, events=None, vectorized=false, args=None, jac=None, jac_sparsity=None, **options))]
pub fn solve_ivp_py<'py>(
    py: Python<'py>,
    fun: Bound<'py, PyAny>,
    t_span: Bound<'py, PyAny>,
    y0: Bound<'py, PyAny>,
    method: Option<Bound<'py, PyAny>>,
    t_eval: Option<Bound<'py, PyAny>>,
    dense_output: bool,
    events: Option<Bound<'py, PyAny>>,
    vectorized: bool,
    args: Option<Bound<'py, PyTuple>>,
    jac: Option<Bound<'py, PyAny>>,
    jac_sparsity: Option<Bound<'py, PyAny>>,
    options: Option<Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = vectorized; // Not currently used

    // Parse inputs
    let (t0, tf) = parse_t_span(&t_span)?;
    let y0_vec = extract_float_array(&y0)?;

    // Parse method
    let method_enum = parse_method(method);

    // Parse t_eval
    let t_eval_vec = parse_t_eval(t_eval)?;

    // Parse events
    let (event_funs, event_configs) = parse_events(&events)?;

    // Parse jac_sparsity
    let sparsity_structure = match jac_sparsity {
        Some(sp) => Some(SparsityStructure::from_python(&sp)?),
        None => None,
    };

    // Parse solver options
    let (rtol, atol, max_step_opt, first_step_opt) = parse_options(&options)?;

    // Build solver options
    let opts = Options::builder()
        .method(method_enum)
        .dense_output(dense_output)
        .maybe_t_eval(t_eval_vec)
        .maybe_max_step(max_step_opt)
        .maybe_first_step(first_step_opt)
        .rtol(rtol)
        .atol(atol)
        .build();

    // Check if Jacobian is a constant matrix (not callable)
    // For scipy compatibility: constant Jacobians should have njev=0
    let is_constant_jac = jac.as_ref().map_or(false, |j| !j.is_callable());

    // Create IVP wrapper
    let python_ivp = PythonIVP::new(fun, event_funs, jac, sparsity_structure, args, event_configs, py);

    // Solve
    let result = solve_ivp(&python_ivp, t0, tf, &y0_vec, opts);

    match result {
        Ok(sol) => build_result(py, sol, events.is_some(), is_constant_jac),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Solver failed: {:?}",
            e
        ))),
    }
}

/// Parse the method argument into a Method enum.
fn parse_method(method: Option<Bound<'_, PyAny>>) -> Method {
    if let Some(m) = method {
        if let Ok(s) = m.extract::<String>() {
            return Method::from(s.as_str());
        }
    }
    Method::DOPRI5
}

/// Parse optional t_eval array.
fn parse_t_eval(t_eval: Option<Bound<'_, PyAny>>) -> PyResult<Option<Vec<Float>>> {
    match t_eval {
        Some(te) => {
            let vec = extract_float_array(&te)?;
            Ok(Some(vec))
        }
        None => Ok(None),
    }
}

/// Parse event functions and extract their configurations.
fn parse_events<'py>(
    events: &Option<Bound<'py, PyAny>>,
) -> PyResult<(Vec<Bound<'py, PyAny>>, Vec<EventConfig>)> {
    let mut event_funs = Vec::new();
    let mut event_configs = Vec::new();

    if let Some(ev) = events {
        // Collect event functions
        if let Ok(lst) = ev.cast::<PyList>() {
            for item in lst.iter() {
                event_funs.push(item.clone());
            }
        } else if let Ok(tup) = ev.extract::<Vec<Bound<'py, PyAny>>>() {
            for item in tup {
                event_funs.push(item);
            }
        } else {
            // Single callable
            event_funs.push(ev.clone());
        }

        // Extract event attributes
        for ef in &event_funs {
            let mut config = EventConfig::new();

            if let Ok(term) = ef.getattr("terminal") {
                if let Ok(is_term) = term.extract::<bool>() {
                    if is_term {
                        config.terminal();
                    }
                }
            }

            if let Ok(dir) = ef.getattr("direction") {
                if let Ok(d) = dir.extract::<f64>() {
                    config.direction(Direction::from(d as i32));
                }
            }

            event_configs.push(config);
        }
    }

    Ok((event_funs, event_configs))
}

/// Parse solver options from kwargs.
fn parse_options(
    options: &Option<Bound<'_, PyDict>>,
) -> PyResult<(Tolerance, Tolerance, Option<Float>, Option<Float>)> {
    let mut rtol: Tolerance = Tolerance::Scalar(1e-3);
    let mut atol: Tolerance = Tolerance::Scalar(1e-6);
    let mut max_step: Option<Float> = None;
    let mut first_step: Option<Float> = None;

    if let Some(opts) = options {
        if let Ok(Some(r)) = opts.get_item("rtol") {
            // Try scalar first, then array
            if let Ok(val) = r.extract::<Float>() {
                rtol = Tolerance::Scalar(val);
            } else if let Ok(arr) = r.extract::<Vec<Float>>() {
                rtol = Tolerance::Vector(arr);
            }
        }
        if let Ok(Some(a)) = opts.get_item("atol") {
            // Try scalar first, then array
            if let Ok(val) = a.extract::<Float>() {
                atol = Tolerance::Scalar(val);
            } else if let Ok(arr) = a.extract::<Vec<Float>>() {
                atol = Tolerance::Vector(arr);
            }
        }
        if let Ok(Some(m)) = opts.get_item("max_step") {
            if let Ok(val) = m.extract::<Float>() {
                max_step = Some(val);
            }
        }
        if let Ok(Some(f)) = opts.get_item("first_step") {
            if let Ok(val) = f.extract::<Float>() {
                first_step = Some(val);
            }
        }
    }

    Ok((rtol, atol, max_step, first_step))
}

/// Build the PyOdeResult from the Rust Solution.
fn build_result<'py>(
    py: Python<'py>,
    sol: crate::solve::Solution,
    has_events: bool,
    is_constant_jac: bool,
) -> PyResult<Bound<'py, PyAny>> {
    // Transpose y from (time, state) to (state, time) for SciPy compatibility
    let n_steps = sol.y.len();
    let n_states = if n_steps > 0 { sol.y[0].len() } else { 0 };

    let mut y_transposed = vec![0.0; n_steps * n_states];
    for (i, step) in sol.y.iter().enumerate() {
        for (j, val) in step.iter().enumerate() {
            y_transposed[j * n_steps + i] = *val;
        }
    }

    let y_arr = PyArray1::from_vec(py, y_transposed).reshape((n_states, n_steps))?;

    // Build t_events list
    let t_events_list = if has_events {
        Some(
            PyList::new(
                py,
                sol.t_events
                    .iter()
                    .map(|te| PyArray1::from_vec(py, te.clone())),
            )?
            .into_any()
            .unbind(),
        )
    } else {
        None
    };

    // Build y_events list
    let y_events_list = if has_events {
        let mut y_events_py = Vec::new();
        for ye in sol.y_events {
            if ye.is_empty() {
                y_events_py.push(PyList::empty(py).into_any());
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

    // Convert status
    let status_int = match sol.status {
        crate::status::Status::Success => 0,
        crate::status::Status::UserInterrupt => 1,
        _ => -1,
    };

    // Build dense output object
    let sol_obj = if let Some(cont) = sol.continuous_sol {
        Some(Py::new(py, PyOdeSolution::new(cont))?)
    } else {
        None
    };

    let result = PyOdeResult {
        t: PyArray1::from_vec(py, sol.t).into_any().unbind(),
        y: y_arr.into_any().unbind(),
        t_events: t_events_list,
        y_events: y_events_list,
        nfev: sol.nfev,
        njev: if is_constant_jac { 0 } else { sol.njev },
        nlu: sol.nlu,
        status: status_int,
        message: format!("{:?}", sol.status),
        success: status_int >= 0,
        sol: sol_obj,
    };

    Ok(Bound::new(py, result)?.into_any())
}
