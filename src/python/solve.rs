//! Main solve_ivp function for Python.
//!
//! This module contains the `solve_ivp` function that serves as the primary
//! entry point for solving ODEs from Python.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::panic::{AssertUnwindSafe, catch_unwind};

use crate::Float;
use crate::methods::{SymplecticMethod, Tolerance};
use crate::solve::event::{Direction, EventConfig};
use crate::solve::{Ivp, JacobianSource, Method};

use super::conversion::{extract_float_array, parse_t_span};
use super::ivp_wrapper::{
    PythonCallbackError, PythonCallbackErrorKind, PythonHamiltonianIVP, PythonIVP,
    PythonSecondOrderIVP,
};
use super::result::PyOdeResult;
use super::solution::PyOdeSolution;
use super::sparsity::SparsityStructure;

/// Solve an initial value problem for a system of ODEs.
///
/// This function numerically integrates a system of ordinary differential
/// equations given an initial value::
///
///     dy / dt = f(t, y, p)
///     y(t0) = y0
///
/// Parameters
/// ----------
/// fun : callable
///     The right-hand side function `f(t, y, p, *args)`.
/// t_span : 2-member sequence
///     Interval of integration (t0, tf).
/// y0 : array_like, shape (n,)
///     Initial state.
/// method : str, optional
///     Integration method to use.
/// t_eval : array_like or None, optional
///     Times at which to store the computed solution.
/// dense_output : bool, optional
///     Whether to compute a continuous solution.
/// events : callable, or list of callables, optional
///     Events to track.
/// args : tuple, optional
///     Additional arguments to pass to the user-defined functions.
/// jac : array_like, callable or None, optional
///     Jacobian matrix of the right-hand side.
/// p : array_like, optional
///     Parameters passed to the system functions.
/// quadrature : callable, optional
///     Function `g(t, y, p, *args)` to integrate over time.
/// forward_sensitivity : bool, optional
///     Whether to compute forward sensitivities dy/dp.
///
/// Returns
/// -------
/// OdeResult object with the solution data.
#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(name = "solve_ivp")]
#[pyo3(signature = (fun, t_span, y0, method=None, t_eval=None, dense_output=false, events=None, vectorized=false, args=None, jac=None, jac_sparsity=None, p=None, quadrature=None, forward_sensitivity=false, **options))]
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
    p: Option<Bound<'py, PyAny>>,
    quadrature: Option<Bound<'py, PyAny>>,
    forward_sensitivity: bool,
    options: Option<Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = vectorized; // Not currently used
    let caught = catch_unwind(AssertUnwindSafe(|| {
        // Parse inputs
        let (t0, tf) = parse_t_span(&t_span)?;
        let y0_vec = extract_float_array(&y0)?;

        // Parse method
        let parsed_method = parse_method(method);

        // Parse t_eval
        let t_eval_vec = parse_t_eval(t_eval)?;

        if matches!(parsed_method, ParsedMethod::Symplectic(_)) && events.is_some() {
            return Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "events are not yet supported for symplectic methods",
            ));
        }

        // Parse events
        let (event_funs, event_configs) = parse_events(&events)?;

        // Parse jac_sparsity
        let sparsity_structure = match jac_sparsity {
            Some(sp) => Some(SparsityStructure::from_python(&sp)?),
            None => None,
        };

        // Parse solver options
        let parsed_options = parse_options(&options)?;

        // Parse p
        let p_vec = match p {
            Some(ref pv) => extract_float_array(pv)?,
            None => Vec::new(),
        };

        let dim_y = y0_vec.len();
        let dim_p = p_vec.len();

        let result = match parsed_method {
            ParsedMethod::Standard(method_enum) => {
                let is_constant_jac = jac.as_ref().is_some_and(|j| !j.is_callable());
                let jacobian_source = jac.as_ref().map(|_| JacobianSource::UserProvided);
                let python_ivp = PythonIVP::new(
                    fun.clone(),
                    event_funs,
                    jac,
                    quadrature,
                    sparsity_structure,
                    args.clone(),
                    event_configs,
                    py,
                );

                if forward_sensitivity {
                    let mut y0_augmented = vec![0.0; dim_y + dim_y * dim_p];
                    y0_augmented[0..dim_y].copy_from_slice(&y0_vec);

                    let sensitivity_system = crate::solve::sensitivity::ForwardSensitivitySystem::new(
                        &python_ivp,
                        dim_y,
                        dim_p,
                    );

                    Ivp::first_order(&sensitivity_system, t0, tf, &y0_augmented)
                        .p(p_vec.clone())
                        .method(method_enum)
                        .dense_output(dense_output)
                        .maybe_t_eval(t_eval_vec)
                        .maybe_max_step(parsed_options.max_step)
                        .maybe_min_step(parsed_options.min_step)
                        .maybe_first_step(parsed_options.first_step)
                        .maybe_max_steps(parsed_options.max_steps)
                        .maybe_jacobian_source(jacobian_source)
                        .rtol(parsed_options.rtol)
                        .atol(parsed_options.atol)
                        .solve()
                        .map(|sol| (sol, events.is_some(), is_constant_jac))
                } else {
                    Ivp::first_order(&python_ivp, t0, tf, &y0_vec)
                        .p(p_vec.clone())
                        .method(method_enum)
                        .dense_output(dense_output)
                        .maybe_t_eval(t_eval_vec)
                        .maybe_max_step(parsed_options.max_step)
                        .maybe_min_step(parsed_options.min_step)
                        .maybe_first_step(parsed_options.first_step)
                        .maybe_max_steps(parsed_options.max_steps)
                        .maybe_jacobian_source(jacobian_source)
                        .rtol(parsed_options.rtol)
                        .atol(parsed_options.atol)
                        .solve()
                        .map(|sol| (sol, events.is_some(), is_constant_jac))
                }
            }
            ParsedMethod::Symplectic(method_enum) => {
                let step_size = parsed_options.step_size.or(parsed_options.first_step);

                let (q0, p0) = split_symplectic_state(&y0_vec)?;

                let symplectic_result = match resolve_symplectic_problem(&fun, &options)? {
                    SymplecticProblem::Hamiltonian {
                        position_derivative,
                        momentum_derivative,
                    } => {
                        let problem = PythonHamiltonianIVP::new(
                            position_derivative,
                            momentum_derivative,
                            args.clone(),
                            py,
                        );
                        Ivp::hamiltonian(&problem, t0, tf, q0, p0)
                            .p(p_vec.clone())
                            .method(method_enum)
                            .maybe_step_size(step_size)
                            .dense_output(dense_output)
                            .maybe_t_eval(t_eval_vec.clone())
                            .maybe_max_steps(parsed_options.max_steps)
                            .solve()
                    }
                    SymplecticProblem::SecondOrder { acceleration } => {
                        let problem = PythonSecondOrderIVP::new(acceleration, args.clone(), py);
                        Ivp::second_order(&problem, t0, tf, q0, p0)
                            .p(p_vec.clone())
                            .method(method_enum)
                            .maybe_step_size(step_size)
                            .dense_output(dense_output)
                            .maybe_t_eval(t_eval_vec)
                            .maybe_max_steps(parsed_options.max_steps)
                            .solve()
                    }
                };

                symplectic_result.map(|sol| (sol, false, false))
            }
        };

        match result {
            Ok((sol, has_events, is_constant_jac)) => {
                build_result(
                    py,
                    sol,
                    has_events,
                    is_constant_jac,
                    dim_y,
                    dim_p,
                    forward_sensitivity,
                    fun.clone().unbind(),
                    args.clone().map(|a| a.unbind()),
                    p.clone().map(|pv| pv.unbind()),
                )
            }
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Solver failed: {:?}",
                e
            ))),
        }
    }));

    match caught {
        Ok(result) => result,
        Err(payload) => Err(panic_payload_to_pyerr(payload)),
    }
}

enum ParsedMethod {
    Standard(Method),
    Symplectic(SymplecticMethod),
}

enum SymplecticProblem<'py> {
    SecondOrder {
        acceleration: Bound<'py, PyAny>,
    },
    Hamiltonian {
        position_derivative: Bound<'py, PyAny>,
        momentum_derivative: Bound<'py, PyAny>,
    },
}

/// Parse the method argument into a standard or symplectic method enum.
fn parse_method(method: Option<Bound<'_, PyAny>>) -> ParsedMethod {
    if let Some(m) = method
        && let Ok(s) = m.extract::<String>()
    {
        let upper = s.to_uppercase();
        return match upper.as_str() {
            "SYMPLECTICEULERKICKDRIFT" | "SEKD" => {
                ParsedMethod::Symplectic(SymplecticMethod::SymplecticEulerKickDrift)
            }
            "SYMPLECTICEULERDRIFTKICK" | "SEDK" => {
                ParsedMethod::Symplectic(SymplecticMethod::SymplecticEulerDriftKick)
            }
            "VELOCITYVERLET" | "VERLET" | "LEAPFROG" => {
                ParsedMethod::Symplectic(SymplecticMethod::VelocityVerlet)
            }
            "RUTH3" => ParsedMethod::Symplectic(SymplecticMethod::Ruth3),
            "YOSHIDA4" => ParsedMethod::Symplectic(SymplecticMethod::Yoshida4),
            _ => ParsedMethod::Standard(Method::from(s.as_str())),
        };
    }
    ParsedMethod::Standard(Method::DOPRI5)
}

fn resolve_symplectic_problem<'py>(
    fun: &Bound<'py, PyAny>,
    options: &Option<Bound<'py, PyDict>>,
) -> PyResult<SymplecticProblem<'py>> {
    if let Some((position_derivative, momentum_derivative)) =
        callable_pair_from_options(options, "position_derivative", "momentum_derivative")?
    {
        return Ok(SymplecticProblem::Hamiltonian {
            position_derivative,
            momentum_derivative,
        });
    }

    if let Some((position_derivative, momentum_derivative)) =
        callable_pair_from_options(options, "drift", "kick")?
    {
        return Ok(SymplecticProblem::Hamiltonian {
            position_derivative,
            momentum_derivative,
        });
    }

    if let Some(acceleration) = callable_option(options, "acceleration")? {
        return Ok(SymplecticProblem::SecondOrder { acceleration });
    }

    if let Some((position_derivative, momentum_derivative)) =
        callable_pair_from_attributes(fun, "position_derivative", "momentum_derivative")?
    {
        return Ok(SymplecticProblem::Hamiltonian {
            position_derivative,
            momentum_derivative,
        });
    }

    if let Some((position_derivative, momentum_derivative)) =
        callable_pair_from_attributes(fun, "drift", "kick")?
    {
        return Ok(SymplecticProblem::Hamiltonian {
            position_derivative,
            momentum_derivative,
        });
    }

    if let Some((position_derivative, momentum_derivative)) = callable_pair_from_sequence(fun)? {
        return Ok(SymplecticProblem::Hamiltonian {
            position_derivative,
            momentum_derivative,
        });
    }

    if fun.is_callable() {
        return Ok(SymplecticProblem::SecondOrder {
            acceleration: fun.clone(),
        });
    }

    if let Some(acceleration) = callable_attribute(fun, "acceleration")? {
        return Ok(SymplecticProblem::SecondOrder { acceleration });
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "symplectic methods require one of: a callable `fun(t, q)` for second-order systems; `fun=(position_derivative, momentum_derivative)` for Hamiltonian systems; `position_derivative=...` and `momentum_derivative=...` keyword callbacks; or a legacy object exposing `acceleration`, `position_derivative`/`momentum_derivative`, or `drift`/`kick`",
    ))
}

fn callable_option<'py>(
    options: &Option<Bound<'py, PyDict>>,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let Some(opts) = options else {
        return Ok(None);
    };

    match opts.get_item(name)? {
        Some(value) if value.is_none() => Ok(None),
        Some(value) if value.is_callable() => Ok(Some(value)),
        Some(_) => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "`{name}` must be callable when provided"
        ))),
        None => Ok(None),
    }
}

fn callable_pair_from_options<'py>(
    options: &Option<Bound<'py, PyDict>>,
    first: &str,
    second: &str,
) -> PyResult<Option<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
    let first_value = callable_option(options, first)?;
    let second_value = callable_option(options, second)?;

    match (first_value, second_value) {
        (Some(first_callable), Some(second_callable)) => {
            Ok(Some((first_callable, second_callable)))
        }
        (None, None) => Ok(None),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "`{first}` and `{second}` must either both be provided or both be omitted"
        ))),
    }
}

fn callable_attribute<'py>(
    object: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    match object.getattr(name) {
        Ok(value) if value.is_none() => Ok(None),
        Ok(value) if value.is_callable() => Ok(Some(value)),
        Ok(_) => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "`fun.{name}` must be callable"
        ))),
        Err(_) => Ok(None),
    }
}

fn callable_pair_from_attributes<'py>(
    object: &Bound<'py, PyAny>,
    first: &str,
    second: &str,
) -> PyResult<Option<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
    let first_value = callable_attribute(object, first)?;
    let second_value = callable_attribute(object, second)?;

    match (first_value, second_value) {
        (Some(first_callable), Some(second_callable)) => {
            Ok(Some((first_callable, second_callable)))
        }
        (None, None) => Ok(None),
        _ => Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "`fun.{first}` and `fun.{second}` must either both be present or both be absent"
        ))),
    }
}

fn callable_pair_from_sequence<'py>(
    object: &Bound<'py, PyAny>,
) -> PyResult<Option<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
    if let Ok(sequence) = object.cast::<PyTuple>() {
        return callable_pair_from_items(sequence.iter().collect(), "`fun` tuple");
    }

    if let Ok(sequence) = object.cast::<PyList>() {
        return callable_pair_from_items(sequence.iter().collect(), "`fun` list");
    }

    Ok(None)
}

fn callable_pair_from_items<'py>(
    items: Vec<Bound<'py, PyAny>>,
    context: &str,
) -> PyResult<Option<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
    if items.is_empty() {
        return Ok(None);
    }

    if items.len() != 2 {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{context} must contain exactly two callables"
        )));
    }

    if !items[0].is_callable() || !items[1].is_callable() {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "{context} must contain only callables"
        )));
    }

    let mut iter = items.into_iter();
    Ok(Some((iter.next().unwrap(), iter.next().unwrap())))
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

            if let Ok(term) = ef.getattr("terminal")
                && let Ok(is_term) = term.extract::<bool>()
                && is_term
            {
                config.terminal();
            }

            if let Ok(dir) = ef.getattr("direction")
                && let Ok(d) = dir.extract::<f64>()
            {
                config.direction(Direction::from(d as i32));
            }

            event_configs.push(config);
        }
    }

    Ok((event_funs, event_configs))
}

/// Parse solver options from kwargs.
struct ParsedOptions {
    rtol: Tolerance,
    atol: Tolerance,
    max_step: Option<Float>,
    min_step: Option<Float>,
    first_step: Option<Float>,
    step_size: Option<Float>,
    max_steps: Option<usize>,
}

fn parse_options(options: &Option<Bound<'_, PyDict>>) -> PyResult<ParsedOptions> {
    let mut parsed = ParsedOptions {
        rtol: Tolerance::Scalar(1e-3),
        atol: Tolerance::Scalar(1e-6),
        max_step: None,
        min_step: None,
        first_step: None,
        step_size: None,
        max_steps: None,
    };

    if let Some(opts) = options {
        if let Ok(Some(r)) = opts.get_item("rtol") {
            if let Ok(val) = r.extract::<Float>() {
                parsed.rtol = Tolerance::Scalar(val);
            } else if let Ok(arr) = r.extract::<Vec<Float>>() {
                parsed.rtol = Tolerance::Vector(arr);
            }
        }
        if let Ok(Some(a)) = opts.get_item("atol") {
            if let Ok(val) = a.extract::<Float>() {
                parsed.atol = Tolerance::Scalar(val);
            } else if let Ok(arr) = a.extract::<Vec<Float>>() {
                parsed.atol = Tolerance::Vector(arr);
            }
        }
        if let Ok(Some(m)) = opts.get_item("max_step")
            && let Ok(val) = m.extract::<Float>()
        {
            parsed.max_step = Some(val);
        }
        if let Ok(Some(m)) = opts.get_item("min_step")
            && let Ok(val) = m.extract::<Float>()
        {
            parsed.min_step = Some(val);
        }
        if let Ok(Some(f)) = opts.get_item("first_step")
            && let Ok(val) = f.extract::<Float>()
        {
            parsed.first_step = Some(val);
        }
        if let Ok(Some(f)) = opts.get_item("step_size")
            && let Ok(val) = f.extract::<Float>()
        {
            parsed.step_size = Some(val);
        }
        if let Ok(Some(ms)) = opts.get_item("max_steps")
            && let Ok(val) = ms.extract::<usize>()
        {
            parsed.max_steps = Some(val);
        }
    }

    Ok(parsed)
}

fn split_symplectic_state(y0: &[Float]) -> PyResult<(&[Float], &[Float])> {
    if y0.is_empty() || !y0.len().is_multiple_of(2) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "symplectic methods require an even-length initial state [q..., p...] or [q..., v...]",
        ));
    }

    let mid = y0.len() / 2;
    Ok(y0.split_at(mid))
}

fn panic_payload_to_pyerr(payload: Box<dyn std::any::Any + Send>) -> PyErr {
    match payload.downcast::<PythonCallbackError>() {
        Ok(callback_error) => match callback_error.kind {
            PythonCallbackErrorKind::Type => {
                pyo3::exceptions::PyTypeError::new_err(callback_error.message)
            }
            PythonCallbackErrorKind::Value => {
                pyo3::exceptions::PyValueError::new_err(callback_error.message)
            }
            PythonCallbackErrorKind::Runtime => {
                pyo3::exceptions::PyRuntimeError::new_err(callback_error.message)
            }
        },
        Err(payload) => match payload.downcast::<String>() {
            Ok(message) => pyo3::exceptions::PyRuntimeError::new_err(*message),
            Err(payload) => match payload.downcast::<&'static str>() {
                Ok(message) => pyo3::exceptions::PyRuntimeError::new_err(*message),
                Err(_) => pyo3::exceptions::PyRuntimeError::new_err("internal solver panic"),
            },
        },
    }
}

/// Build the PyOdeResult from the Rust Solution.
#[allow(clippy::too_many_arguments)]
fn build_result<'py>(
    py: Python<'py>,
    sol: crate::solve::Solution,
    has_events: bool,
    is_constant_jac: bool,
    dim_y: usize,
    dim_p: usize,
    has_sensitivity: bool,
    fun: Py<PyAny>,
    args: Option<Py<PyTuple>>,
    p: Option<Py<PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let n_steps = sol.t.len();

    // Split y into state and sensitivities if necessary
    let (y_out, s_out) = if has_sensitivity && dim_p > 0 {
        let mut y_vals = vec![0.0; n_steps * dim_y];
        let mut s_vals = vec![0.0; n_steps * dim_y * dim_p];

        for (i, step_y) in sol.y.iter().enumerate() {
            // State
            for j in 0..dim_y {
                y_vals[j * n_steps + i] = step_y[j];
            }
            // Sensitivities
            for j in 0..(dim_y * dim_p) {
                s_vals[j * n_steps + i] = step_y[dim_y + j];
            }
        }
        (
            PyArray1::from_vec(py, y_vals).reshape((dim_y, n_steps))?.into_any().unbind(),
            Some(PyArray1::from_vec(py, s_vals).reshape((dim_p, dim_y, n_steps))?.into_any().unbind()),
        )
    } else {
        let n_states = if n_steps > 0 { sol.y[0].len() } else { 0 };
        let mut y_transposed = vec![0.0; n_steps * n_states];
        for (i, step) in sol.y.iter().enumerate() {
            for (j, val) in step.iter().enumerate() {
                y_transposed[j * n_steps + i] = *val;
            }
        }
        (
            PyArray1::from_vec(py, y_transposed).reshape((n_states, n_steps))?.into_any().unbind(),
            None,
        )
    };

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

    // Build quad results
    let quad_arr = if !sol.quad.is_empty() {
        Some(PyArray1::from_vec(py, sol.quad).into_any().unbind())
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
        fun,
        args,
        p,
        t: PyArray1::from_vec(py, sol.t).into_any().unbind(),
        y: y_out,
        s: s_out,
        t_events: t_events_list,
        y_events: y_events_list,
        quad: quad_arr,
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
