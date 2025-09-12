//! Linear solves: A x = b via LU with partial pivoting.

use crate::Float;

use super::base::{Matrix, MatrixStorage};

impl Matrix {
    /// Solve A x = b, returning x. 
    pub fn lin_solve(&self, b: &[Float]) -> Vec<Float> {
        let mut b_copy = b.to_vec();
        self.lin_solve_mut(&mut b_copy);
        b_copy
    }

    /// In-place solve: overwrites `b` with `x`.
    pub fn lin_solve_mut(&self, b: &mut [Float]) {
        let n = self.n;
        assert_eq!(
            b.len(),
            n,
            "dimension mismatch in solve: A is {}x{}, b has length {}",
            n,
            n,
            b.len()
        );

        // Densify A into row-major Vec<T>
        let mut a = vec![0.0; n * n];
        match &self.storage {
            MatrixStorage::Identity => {
                for i in 0..n {
                    a[i * n + i] = 1.0;
                }
            }
            MatrixStorage::Full => {
                a.copy_from_slice(&self.data[0..n * n]);
            }
            MatrixStorage::Banded { ml, mu, .. } => {
                let rows = *ml + *mu + 1;
                for j in 0..self.m {
                    for r in 0..rows {
                        let k = r as isize - *mu as isize;
                        let i_signed = j as isize + k;
                        if i_signed >= 0 && (i_signed as usize) < self.n {
                            let i = i_signed as usize;
                            a[i * self.m + j] += self.data[r * self.m + j];
                        }
                    }
                }
            }
        }

        // LU with partial pivoting, applying permutations to b
        for k in 0..n {
            // pivot
            let mut pivot_row = k;
            let mut pivot_val = a[k * n + k].abs();
            for i in (k + 1)..n {
                let val = a[i * n + k].abs();
                if val > pivot_val {
                    pivot_val = val;
                    pivot_row = i;
                }
            }
            if pivot_val == 0.0 {
                panic!("singular matrix in solve");
            }
            if pivot_row != k {
                for j in 0..n {
                    a.swap(k * n + j, pivot_row * n + j);
                }
                b.swap(k, pivot_row);
            }
            // Eliminate below the pivot
            let akk = a[k * n + k];
            for i in (k + 1)..n {
                let factor = a[i * n + k] / akk;
                a[i * n + k] = factor;
                for j in (k + 1)..n {
                    a[i * n + j] = a[i * n + j] - factor * a[k * n + j];
                }
            }
        }

        // Forward solve Ly = Pb (b is permuted)
        for i in 0..n {
            let mut sum = b[i];
            for k in 0..i {
                sum -= a[i * n + k] * b[k];
            }
            b[i] = sum;
        }
        // Backward solve Ux = y
        for i in (0..n).rev() {
            let mut sum = b[i];
            for k in (i + 1)..n {
                sum -= a[i * n + k] * b[k];
            }
            b[i] = sum / a[i * n + i];
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;

    #[test]
    fn solve_full_2x2() {
        // A = [[3, 2],[1, 4]], b = [5, 6] -> x = [0.8, 1.3]
        let mut a: Matrix = Matrix::full(2, 2);
        a[(0, 0)] = 3.0;
        a[(0, 1)] = 2.0;
        a[(1, 0)] = 1.0;
        a[(1, 1)] = 4.0;
        let b = vec![5.0, 6.0];
        let x = a.lin_solve(&b);
        // Solve manually: [[3,2],[1,4]] x = [5,6] => x = [ (20-12)/10, (15-5)/10 ] = [0.8, 1.3]
        assert!((x[0] - 0.8).abs() < 1e-12);
        assert!((x[1] - 1.3).abs() < 1e-12);
    }
}
