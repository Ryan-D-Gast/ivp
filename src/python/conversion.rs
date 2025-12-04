//! Type conversion utilities between Python and Rust.
//!
//! Provides helper functions for converting between Python types (numpy arrays,
//! lists, tuples) and Rust types (Vec, slices).

use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::Float;

/// Extract a vector of floats from various Python array-like types.
///
/// Supports:
/// - numpy arrays (float64, int64, int32)
/// - Python lists
pub fn extract_float_array(obj: &Bound<'_, PyAny>) -> PyResult<Vec<Float>> {
    // Try float64 numpy array first (most common)
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<Float>>() {
        return Ok(arr.as_slice()?.to_vec());
    }

    // Handle integer numpy arrays
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<i64>>() {
        return Ok(arr.as_slice()?.iter().map(|&x| x as Float).collect());
    }
    if let Ok(arr) = obj.extract::<PyReadonlyArray1<i32>>() {
        return Ok(arr.as_slice()?.iter().map(|&x| x as Float).collect());
    }

    // Try Python list
    if let Ok(lst) = obj.cast::<PyList>() {
        return lst
            .iter()
            .map(|x| x.extract::<Float>())
            .collect::<Result<Vec<_>, _>>();
    }

    Err(pyo3::exceptions::PyValueError::new_err(
        "Expected array_like (numpy array or list)",
    ))
}

/// Parse a 2-element tuple or list into (t0, tf).
pub fn parse_t_span(t_span: &Bound<'_, PyAny>) -> PyResult<(Float, Float)> {
    if let Ok(tup) = t_span.extract::<(Float, Float)>() {
        return Ok(tup);
    }
    if let Ok(lst) = t_span.extract::<Vec<Float>>() {
        if lst.len() != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "t_span must have exactly 2 elements",
            ));
        }
        return Ok((lst[0], lst[1]));
    }
    Err(pyo3::exceptions::PyValueError::new_err(
        "t_span must be a tuple or list of 2 floats",
    ))
}
