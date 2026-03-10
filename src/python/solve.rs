//! Main solve_ivp function for Python.
//!
//! This module contains the `solve_ivp` function that serves as the primary
//! entry point for solving ODEs from Python.

use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::panic::{catch_unwind, AssertUnwindSafe};

use crate::methods::{SymplecticMethod, Tolerance};
use crate::solve::event::{Direction, EventConfig};
use crate::solve::{Ivp, Method};
use crate::Float;

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
///     dy / dt = f(t, y)
///     y(t0) = y0
///
/// Here t is a 1-D independent variable (time), y(t) is an N-D vector-valued
/// function (state), and an N-D vector-valued function f(t, y) determines the
/// differential equations.
///
/// Parameters
/// ----------
/// fun : callable, tuple of callables, or object
///     For standard first-order methods, ``fun`` is the right-hand side
///     ``fun(t, y)`` and must return an array-like object with the same shape
///     as ``y``.
///
///     For symplectic second-order methods, ``fun`` may be the acceleration
///     callback ``fun(t, q)`` unless ``acceleration=...`` is provided
///     explicitly.
///
///     For symplectic Hamiltonian methods, ``fun`` may be a two-tuple
///     ``(position_derivative, momentum_derivative)`` unless the equivalent
///     keyword callbacks are provided explicitly.
///
///     Legacy objects exposing methods such as ``acceleration(...)``,
///     ``position_derivative(...)``/``momentum_derivative(...)``, or
///     ``drift(...)``/``kick(...)`` are also accepted.
/// t_span : 2-member sequence
///     Interval of integration (t0, tf). The solver starts with t=t0 and
///     integrates until it reaches t=tf. Both t0 and tf must be floats.
/// y0 : array_like, shape (n,)
///     Initial state.
/// method : str, optional
///     Integration method to use:
///
///     * 'RK45' or 'DOPRI5' (default): Explicit Runge-Kutta method of order 5(4).
///       The error is controlled assuming accuracy of the fourth-order method,
///       but steps are taken using the fifth-order accurate formula.
///     * 'RK23': Explicit Runge-Kutta method of order 3(2).
///     * 'DOP853': Explicit Runge-Kutta method of order 8.
///     * 'Radau': Implicit Runge-Kutta method of the Radau IIA family of order 5.
///       Suitable for stiff problems.
///     * 'BDF': Implicit multi-step variable-order (1 to 5) method based on a
///       backward differentiation formula. Suitable for stiff problems.
///     * 'RK4': Classic explicit Runge-Kutta method of order 4 with fixed step size.
///     * 'SymplecticEulerKickDrift', 'SymplecticEulerDriftKick', 'VelocityVerlet',
///       'Ruth3', 'Yoshida4': Fixed-step symplectic methods. For second-order
///       systems, ``fun`` may be a plain callable ``fun(t, q)`` returning the
///       acceleration. For separable Hamiltonian systems, pass either
///       ``fun=(position_derivative, momentum_derivative)`` or provide
///       ``position_derivative=...`` and ``momentum_derivative=...`` as keyword
///       arguments. Legacy object methods ``acceleration(...)``,
///       ``position_derivative(...)``/``momentum_derivative(...)``, and
///       ``drift(...)``/``kick(...)`` are still accepted. In Python, ``y0`` is
///       still the flattened state ``[q..., v...]`` or ``[q..., p...]``.
///
/// t_eval : array_like or None, optional
///     Times at which to store the computed solution, must be sorted and lie
///     within t_span. If None (default), use points selected by the solver.
/// dense_output : bool, optional
///     Whether to compute a continuous solution. Default is False.
/// events : callable, or list of callables, optional
///     Events to track. Each event function has the signature ``event(t, y)``
///     and returns a float. A zero crossing of this function is detected.
///     Event functions can have the following attributes:
///
///     * terminal: bool, whether to terminate integration when this event occurs.
///     * direction: float, direction of a zero crossing. +1 for increasing,
///       -1 for decreasing, 0 for both directions.
///
/// vectorized : bool, optional
///     This argument is provided for scipy compatibility and is currently ignored.
/// args : tuple, optional
///     Additional arguments to pass to the user-defined functions (fun, events, jac).
/// jac : array_like, callable or None, optional
///     Jacobian matrix of the right-hand side with respect to y, required for
///     stiff solvers (Radau, BDF). The Jacobian matrix has shape (n, n) and
///     element (i, j) is ``d f_i / d y_j``.
///     If callable, the signature is ``jac(t, y)``.
///     If array_like, the Jacobian is assumed to be constant.
/// jac_sparsity : array_like, sparse matrix, or None, optional
///     Defines the sparsity structure of the Jacobian matrix for BDF method.
///
/// Returns
/// -------
/// Bunch object with the following fields:
///
/// t : ndarray, shape (n_points,)
///     Time points.
/// y : ndarray, shape (n, n_points)
///     Values of the solution at t.
/// sol : OdeSolution or None
///     Found solution as OdeSolution instance; None if dense_output was False.
/// t_events : list of ndarray or None
///     Contains for each event type a list of arrays at which an event of
///     that type was detected. None if events was None.
/// y_events : list of ndarray or None
///     For each event type, a list of arrays with the state at each event time.
///     None if events was None.
/// nfev : int
///     Number of evaluations of the right-hand side.
/// njev : int
///     Number of evaluations of the Jacobian.
/// nlu : int
///     Number of LU decompositions.
/// status : int
///     Reason for algorithm termination:
///
///     * -1: Integration step failed.
///     *  0: The solver successfully reached the end of t_span.
///     *  1: A termination event occurred.
///
/// message : str
///     Human-readable description of the termination reason.
/// success : bool
///     True if the solver reached the end of t_span or a termination event occurred.
///
/// Other Parameters
/// ----------------
/// step_size : float, optional
///     Fixed step size for symplectic methods. If omitted, ``first_step`` is
///     used as a fallback, and if that is also omitted the step defaults to
///     ``(tf - t0) / 100``.
/// acceleration : callable, optional
///     Explicit acceleration callback for second-order symplectic methods.
///     Use this when you prefer not to pass the acceleration as ``fun``.
/// position_derivative : callable, optional
///     Position derivative callback ``q' = dT/dp`` for Hamiltonian symplectic
///     methods.
/// momentum_derivative : callable, optional
///     Momentum derivative callback ``p' = -dV/dq`` for Hamiltonian symplectic
///     methods.
/// first_step : float, optional
///     Initial step size. Default is determined automatically.
/// max_step : float, optional
///     Maximum allowed step size. Default is inf.
/// min_step : float, optional
///     Minimum allowed step size for stiff solvers (Radau, BDF). Default is 0.
/// max_steps : int, optional
///     Maximum number of steps the solver can take. Default is unlimited.
/// rtol : float, optional
///     Relative tolerance. Default is 1e-3.
/// atol : float, optional
///     Absolute tolerance. Default is 1e-6.
///
/// Callback Requirements
/// ---------------------
/// Standard first-order methods expect ``fun(t, y)`` to return a one-dimensional
/// array-like object with the same length as ``y``.
///
/// Symplectic second-order methods expect either:
///
/// * ``fun(t, q)``
/// * ``acceleration=...``
///
/// Both forms must return a one-dimensional array-like object with length
/// ``len(y0) // 2``.
/// The flattened initial state must be ordered as ``[q..., v...]``. For example,
/// a 3D second-order system uses ``y0 = [x, y, z, vx, vy, vz]``, while the
/// callback still receives only ``q = [x, y, z]``.
///
/// Symplectic Hamiltonian methods expect either:
///
/// * ``fun=(position_derivative, momentum_derivative)``
/// * ``position_derivative=...`` and ``momentum_derivative=...``
///
/// Each Hamiltonian callback must return a one-dimensional array-like object
/// with length ``len(y0) // 2``.
///
/// Common Errors
/// -------------
/// Misconfigured callbacks raise Python exceptions instead of causing a Rust panic.
/// Typical mistakes include returning the wrong number of values, returning a
/// non-numeric result, passing an odd-length ``y0`` to a symplectic method, or
/// providing only one Hamiltonian callback instead of both.
///
/// Examples
/// --------
/// Solve an exponential decay ODE::
///
///     >>> from ivp import solve_ivp
///     >>> import numpy as np
///     >>> def exponential_decay(t, y):
///     ...     return -0.5 * y
///     >>> sol = solve_ivp(exponential_decay, (0, 10), [2, 4, 8])
///     >>> print(sol.t)
///     >>> print(sol.y)
///
/// See Also
/// --------
/// scipy.integrate.solve_ivp : SciPy's equivalent function
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

        let result = match parsed_method {
            ParsedMethod::Standard(method_enum) => {
                let is_constant_jac = jac.as_ref().is_some_and(|j| !j.is_callable());
                let python_ivp = PythonIVP::new(
                    fun,
                    event_funs,
                    jac,
                    sparsity_structure,
                    args,
                    event_configs,
                    py,
                );
                Ivp::first_order(&python_ivp, t0, tf, &y0_vec)
                    .method(method_enum)
                    .dense_output(dense_output)
                    .maybe_t_eval(t_eval_vec)
                    .maybe_max_step(parsed_options.max_step)
                    .maybe_min_step(parsed_options.min_step)
                    .maybe_first_step(parsed_options.first_step)
                    .maybe_max_steps(parsed_options.max_steps)
                    .rtol(parsed_options.rtol)
                    .atol(parsed_options.atol)
                    .solve()
                    .and_then(|sol| Ok((sol, events.is_some(), is_constant_jac)))
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
                            args,
                            py,
                        );
                        Ivp::hamiltonian(&problem, t0, tf, q0, p0)
                            .method(method_enum)
                            .maybe_step_size(step_size)
                            .dense_output(dense_output)
                            .maybe_t_eval(t_eval_vec.clone())
                            .maybe_max_steps(parsed_options.max_steps)
                            .solve()
                    }
                    SymplecticProblem::SecondOrder { acceleration } => {
                        let problem = PythonSecondOrderIVP::new(acceleration, args, py);
                        Ivp::second_order(&problem, t0, tf, q0, p0)
                            .method(method_enum)
                            .maybe_step_size(step_size)
                            .dense_output(dense_output)
                            .maybe_t_eval(t_eval_vec)
                            .maybe_max_steps(parsed_options.max_steps)
                            .solve()
                    }
                };

                symplectic_result.and_then(|sol| Ok((sol, false, false)))
            }
        };

        match result {
            Ok((sol, has_events, is_constant_jac)) => {
                build_result(py, sol, has_events, is_constant_jac)
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
    if let Some(m) = method {
        if let Ok(s) = m.extract::<String>() {
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
        if let Ok(Some(m)) = opts.get_item("max_step") {
            if let Ok(val) = m.extract::<Float>() {
                parsed.max_step = Some(val);
            }
        }
        if let Ok(Some(m)) = opts.get_item("min_step") {
            if let Ok(val) = m.extract::<Float>() {
                parsed.min_step = Some(val);
            }
        }
        if let Ok(Some(f)) = opts.get_item("first_step") {
            if let Ok(val) = f.extract::<Float>() {
                parsed.first_step = Some(val);
            }
        }
        if let Ok(Some(f)) = opts.get_item("step_size") {
            if let Ok(val) = f.extract::<Float>() {
                parsed.step_size = Some(val);
            }
        }
        if let Ok(Some(ms)) = opts.get_item("max_steps") {
            if let Ok(val) = ms.extract::<usize>() {
                parsed.max_steps = Some(val);
            }
        }
    }

    Ok(parsed)
}

fn split_symplectic_state(y0: &[Float]) -> PyResult<(&[Float], &[Float])> {
    if y0.is_empty() || y0.len() % 2 != 0 {
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
