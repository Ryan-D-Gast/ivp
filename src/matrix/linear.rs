//! Linear system solvers for LU-decomposed matrices.
//!
//! This module provides solvers for linear systems that have already been LU-decomposed
//! with partial pivoting.
//!
//! # References
//! - Hairer, E., & Wanner, G. (1996). Solving Ordinary Differential Equations II:
//!   Stiff and Differential-Algebraic Problems. Springer.

use crate::Float;

use super::base::Matrix;

/// Solves a real linear system A*X = B using LU decomposition with partial pivoting.
///
/// The matrix `a` must have been previously decomposed using LU factorization with
/// partial pivoting, and `ip` contains the pivot indices from that decomposition.
///
/// # Arguments
/// * `a` - The LU-decomposed matrix (L and U stored in the same matrix)
/// * `b` - On input: right-hand side vector; On output: solution vector X
/// * `ip` - Pivot indices from the LU decomposition
///
/// # Algorithm
/// The solver performs two phases:
/// 1. **Forward elimination**: Applies the row permutations and solves Ly = Pb
/// 2. **Back substitution**: Solves Ux = y
///
/// Where P is the permutation matrix encoded by the pivot vector `ip`.
///
/// # Panics
/// Panics if the matrix dimensions are inconsistent or if indices are out of bounds.
///
/// # Examples
/// ```rust,ignore
/// use ivp::matrix::{Matrix, lu_decomp, lin_solve};
///
/// let mut a = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
/// let mut ip = [0; 2];
/// lu_decomp(&mut a, &mut ip).unwrap();
/// let mut b = vec![1.0, 2.0];
/// lin_solve(&a, &mut b, &ip);
/// // b now contains the solution
/// ```
///
/// # Mathematical Background
/// For a system Ax = b where A has been factorized as PA = LU:
/// - P is a permutation matrix (represented by pivot indices)
/// - L is unit lower triangular
/// - U is upper triangular
///
/// The solution process is:
/// 1. Solve Ly = Pb (forward substitution with pivoting)
/// 2. Solve Ux = y (back substitution)
pub fn lin_solve(a: &Matrix, b: &mut [Float], ip: &[usize]) {
    let n = a.nrows();
    assert_eq!(b.len(), n, "RHS length must match matrix size");

    // Handle trivial case
    if n == 1 {
        b[0] = b[0] / a[(0, 0)];
        return;
    }

    let nm1 = n - 1;

    // Forward elimination with partial pivoting
    for k in 0..nm1 {
        let kp1 = k + 1;
        let m = ip[k]; // Pivot row index

        // Apply row permutation (swap b[m] and b[k])
        let t = b[m];
        b[m] = b[k];
        b[k] = t;

        // Forward substitution step: b[i] += L[i,k] * b[k]
        for i in kp1..n {
            b[i] = b[i] + a[(i, k)] * t;
        }
    }

    // Back substitution: solve Ux = y
    for kb in 1..n {
        let k = n - kb;

        // Divide by diagonal element
        b[k] = b[k] / a[(k, k)];
        let t = -b[k];

        // Back substitution step: b[i] += U[i,k] * (-b[k])
        for i in 0..k {
            b[i] = b[i] + a[(i, k)] * t;
        }
    }

    // Final division for the first element
    b[0] = b[0] / a[(0, 0)];
}

/// Solves a complex linear system (AR + i*AI)*X = (BR + i*BI) using LU decomposition.
///
/// The complex matrix (AR + i*AI) must have been previously decomposed using complex
/// LU factorization with partial pivoting, and `ip` contains the pivot indices.
///
/// # Arguments
/// * `ar` - Real part of the LU-decomposed complex matrix
/// * `ai` - Imaginary part of the LU-decomposed complex matrix  
/// * `br` - On input: real part of RHS; On output: real part of solution
/// * `bi` - On input: imaginary part of RHS; On output: imaginary part of solution
/// * `ip` - Pivot indices from the complex LU decomposition
///
/// # Algorithm
/// Similar to the real case, but with complex arithmetic:
/// 1. **Forward elimination**: Applies permutations and complex forward substitution
/// 2. **Back substitution**: Complex back substitution with division
///
/// All complex operations are performed using separate real and imaginary parts:
/// - Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
/// - Complex division: (a + bi)/(c + di) = [(ac + bd) + (bc - ad)i]/(c² + d²)
///
/// # Panics
/// Panics if matrix dimensions are inconsistent or if indices are out of bounds.
///
/// # Examples
/// ```rust,ignore
/// use ivp::matrix::{Matrix, lu_decomp_complex, lin_solve_complex};
///
/// let mut ar = Matrix::from_vec(2, 2, vec![1.0, 0.0, 0.0, 1.0]);
/// let mut ai = Matrix::from_vec(2, 2, vec![0.0, 1.0, 1.0, 0.0]);
/// let mut ip = [0; 2];
/// lu_decomp_complex(&mut ar, &mut ai, &mut ip).unwrap();
/// let mut br = vec![1.0, 2.0];
/// let mut bi = vec![0.5, 1.5];
/// lin_solve_complex(&ar, &ai, &mut br, &mut bi, &ip);
/// // br, bi now contain the complex solution
/// ```
///
/// # Mathematical Background
/// For a complex system (A + iB)x = (c + id), the LU factorization gives:
/// P(A + iB) = L + iM + N + iO
/// where the solver handles the complex arithmetic explicitly.
pub fn lin_solve_complex(
    ar: &Matrix,
    ai: &Matrix,
    br: &mut [Float],
    bi: &mut [Float],
    ip: &[usize],
) {
    let n = ar.nrows();
    assert_eq!(br.len(), n, "RHS length must match matrix size");
    assert_eq!(bi.len(), n, "RHS length must match matrix size");

    // Handle trivial case with complex division
    if n == 1 {
        // Complex division: (br + i*bi) / (ar + i*ai)
        let den = ar[(0, 0)] * ar[(0, 0)] + ai[(0, 0)] * ai[(0, 0)];
        let temp_r = (br[0] * ar[(0, 0)] + bi[0] * ai[(0, 0)]) / den;
        let temp_i = (bi[0] * ar[(0, 0)] - br[0] * ai[(0, 0)]) / den;
        br[0] = temp_r;
        bi[0] = temp_i;
        return;
    }

    let nm1 = n - 1;

    // Forward elimination with partial pivoting (complex version)
    for k in 0..nm1 {
        let kp1 = k + 1;
        let m = ip[k]; // Pivot row index

        // Apply row permutation to both real and imaginary parts
        let tr = br[m];
        let ti = bi[m];
        let brk = br[k];
        let bik = bi[k];
        br[m] = brk;
        bi[m] = bik;
        br[k] = tr;
        bi[k] = ti;

        // Complex forward substitution: b[i] += L[i,k] * t
        for i in kp1..n {
            // Complex multiplication: (ar[i,k] + i*ai[i,k]) * (tr + i*ti)
            let prod_r = ar[(i, k)] * tr - ai[(i, k)] * ti;
            let prod_i = ai[(i, k)] * tr + ar[(i, k)] * ti;
            br[i] = br[i] + prod_r;
            bi[i] = bi[i] + prod_i;
        }
    }

    // Complex back substitution
    for kb in 1..n {
        let k = n - kb;

        // Complex division: b[k] = b[k] / a[k,k]
        let den = ar[(k, k)] * ar[(k, k)] + ai[(k, k)] * ai[(k, k)];
        let temp_r = (br[k] * ar[(k, k)] + bi[k] * ai[(k, k)]) / den;
        let temp_i = (bi[k] * ar[(k, k)] - br[k] * ai[(k, k)]) / den;
        br[k] = temp_r;
        bi[k] = temp_i;

        // Prepare for back substitution: t = -b[k]
        let tr = -br[k];
        let ti = -bi[k];

        // Complex back substitution step: b[i] += a[i,k] * t
        for i in 0..k {
            let prod_r = ar[(i, k)] * tr - ai[(i, k)] * ti;
            let prod_i = ai[(i, k)] * tr + ar[(i, k)] * ti;
            br[i] = br[i] + prod_r;
            bi[i] = bi[i] + prod_i;
        }
    }

    // Final complex division for the first element
    let den = ar[(0, 0)] * ar[(0, 0)] + ai[(0, 0)] * ai[(0, 0)];
    let temp_r = (br[0] * ar[(0, 0)] + bi[0] * ai[(0, 0)]) / den;
    let temp_i = (bi[0] * ar[(0, 0)] - br[0] * ai[(0, 0)]) / den;
    br[0] = temp_r;
    bi[0] = temp_i;
}
