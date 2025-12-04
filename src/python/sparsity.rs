//! Sparse Jacobian computation utilities.
//!
//! Implements efficient sparse finite differences using column grouping (graph coloring).
//! When the Jacobian is known to be sparse, columns that don't share any non-zero rows
//! can be perturbed simultaneously, reducing the number of function evaluations.

use numpy::PyReadonlyArray2;
use pyo3::prelude::*;

use crate::Float;

/// Sparsity structure for efficient Jacobian computation.
///
/// Contains the sparsity pattern (which elements are non-zero) and the
/// column grouping (which columns can be computed together).
#[derive(Clone)]
pub struct SparsityStructure {
    /// For each column, list of row indices where the Jacobian is non-zero.
    pub col_to_rows: Vec<Vec<usize>>,
    /// Group assignment for each column.
    pub groups: Vec<usize>,
    /// Number of groups.
    pub n_groups: usize,
    /// Dimension of the Jacobian matrix.
    pub n: usize,
}

impl SparsityStructure {
    /// Create a sparsity structure from a Python sparse matrix.
    pub fn from_python<'py>(sparsity: &Bound<'py, PyAny>) -> PyResult<Self> {
        // Convert to CSC format for efficient column access
        let csc = if let Ok(tocsc) = sparsity.getattr("tocsc") {
            tocsc.call0()?
        } else {
            sparsity.clone()
        };

        // Get shape
        let shape = csc.getattr("shape")?;
        let shape_tuple: (usize, usize) = shape.extract()?;
        let n = shape_tuple.0;
        assert_eq!(n, shape_tuple.1, "Jacobian sparsity must be square");

        // Get indices and indptr from CSC format
        let indices = csc.getattr("indices")?;
        let indptr = csc.getattr("indptr")?;

        let indices_vec: Vec<usize> = if let Ok(arr) = indices.extract::<PyReadonlyArray2<i64>>() {
            arr.as_slice()?.iter().map(|&x| x as usize).collect()
        } else if let Ok(arr) = indices.extract::<Vec<i64>>() {
            arr.iter().map(|&x| x as usize).collect()
        } else if let Ok(arr) = indices.extract::<Vec<i32>>() {
            arr.iter().map(|&x| x as usize).collect()
        } else {
            // Try numpy array with different dtype
            let np = sparsity.py().import("numpy")?;
            let arr = np.call_method1("asarray", (indices,))?;
            let arr = arr.call_method1("astype", ("int64",))?;
            arr.extract::<Vec<i64>>()?.iter().map(|&x| x as usize).collect()
        };

        let indptr_vec: Vec<usize> = if let Ok(arr) = indptr.extract::<PyReadonlyArray2<i64>>() {
            arr.as_slice()?.iter().map(|&x| x as usize).collect()
        } else if let Ok(arr) = indptr.extract::<Vec<i64>>() {
            arr.iter().map(|&x| x as usize).collect()
        } else if let Ok(arr) = indptr.extract::<Vec<i32>>() {
            arr.iter().map(|&x| x as usize).collect()
        } else {
            let np = sparsity.py().import("numpy")?;
            let arr = np.call_method1("asarray", (indptr,))?;
            let arr = arr.call_method1("astype", ("int64",))?;
            arr.extract::<Vec<i64>>()?.iter().map(|&x| x as usize).collect()
        };

        // Build col_to_rows mapping
        let mut col_to_rows = Vec::with_capacity(n);
        for col in 0..n {
            let start = indptr_vec[col];
            let end = indptr_vec[col + 1];
            col_to_rows.push(indices_vec[start..end].to_vec());
        }

        // Compute column groups using greedy coloring
        let (groups, n_groups) = group_columns(&col_to_rows, n);

        Ok(SparsityStructure {
            col_to_rows,
            groups,
            n_groups,
            n,
        })
    }

    /// Columns that belong to a specific group.
    pub fn columns_in_group(&self, group: usize) -> Vec<usize> {
        self.groups
            .iter()
            .enumerate()
            .filter(|&(_, &g)| g == group)
            .map(|(col, _)| col)
            .collect()
    }
}

/// Greedy column grouping algorithm.
///
/// Two columns can be in the same group if they don't share any non-zero rows.
/// This is equivalent to graph coloring where vertices are columns and edges
/// connect columns that share a non-zero row.
fn group_columns(col_to_rows: &[Vec<usize>], n: usize) -> (Vec<usize>, usize) {
    let mut groups = vec![usize::MAX; n];
    let mut n_groups = 0;

    // For each group, track which rows are "used" (have a column assigned)
    let mut group_rows: Vec<Vec<bool>> = Vec::new();

    for col in 0..n {
        let rows = &col_to_rows[col];

        // Find the first group where this column can fit
        // (none of its rows are already used by another column in that group)
        let mut assigned_group = None;
        for (group, used_rows) in group_rows.iter().enumerate() {
            let can_use = rows.iter().all(|&row| !used_rows[row]);
            if can_use {
                assigned_group = Some(group);
                break;
            }
        }

        match assigned_group {
            Some(group) => {
                groups[col] = group;
                // Mark rows as used
                for &row in rows {
                    group_rows[group][row] = true;
                }
            }
            None => {
                // Create a new group
                let new_group = n_groups;
                n_groups += 1;
                groups[col] = new_group;
                let mut used_rows = vec![false; n];
                for &row in rows {
                    used_rows[row] = true;
                }
                group_rows.push(used_rows);
            }
        }
    }

    (groups, n_groups)
}

/// Compute sparse Jacobian using finite differences with column grouping.
///
/// This function evaluates the ODE function `n_groups` times instead of `n` times,
/// where `n_groups` is typically much smaller than `n` for sparse Jacobians.
pub fn sparse_jacobian_fd<F>(
    ode: F,
    x: Float,
    y: &[Float],
    f0: &[Float],
    sparsity: &SparsityStructure,
    j: &mut crate::matrix::Matrix,
) where
    F: Fn(Float, &[Float], &mut [Float]),
{
    let n = sparsity.n;
    let eps = Float::EPSILON.sqrt();

    // For each group, perturb all columns in the group simultaneously
    for group in 0..sparsity.n_groups {
        let cols = sparsity.columns_in_group(group);
        if cols.is_empty() {
            continue;
        }

        // Perturb all columns in this group
        let mut y_perturbed = y.to_vec();
        let mut h = vec![0.0; n];
        for &col in &cols {
            let perturbation = eps * y[col].abs().max(1.0);
            y_perturbed[col] = y[col] + perturbation;
            h[col] = perturbation;
        }

        // Evaluate ODE with all perturbations
        let mut f_perturbed = vec![0.0; n];
        ode(x, &y_perturbed, &mut f_perturbed);

        // Extract Jacobian columns from the difference
        for &col in &cols {
            let perturbation = h[col];
            // Only update rows that are non-zero in this column
            for &row in &sparsity.col_to_rows[col] {
                j[(row, col)] = (f_perturbed[row] - f0[row]) / perturbation;
            }
        }
    }
}
