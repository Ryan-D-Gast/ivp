//! Matrix multiplication.

use crate::Float;

use super::base::{Matrix, MatrixStorage};

// Matrix * scalar
impl Matrix {
    /// Return a new matrix where each stored entry is multiplied by `rhs`.
    pub fn component_mul(mut self, rhs: Float) -> Self {
        match &mut self.storage {
            MatrixStorage::Identity => Matrix::diagonal(vec![rhs; self.n]),
            MatrixStorage::Full => {
                for v in &mut self.data {
                    *v *= rhs;
                }
                self
            }
            MatrixStorage::Banded { ml, mu, .. } => {
                let n = self.n;
                let data = self.data.into_iter().map(|x| x * rhs).collect();
                Matrix {
                    n,
                    m: n,
                    data,
                    storage: MatrixStorage::Banded {
                        ml: *ml,
                        mu: *mu,
                    },
                }
            }
        }
    }

    /// In-place component-wise scalar multiplication: self[i,j] *= rhs for all stored entries.
    /// For Identity, converts to a diagonal banded matrix with `rhs` on the diagonal.
    pub fn component_mul_mut(&mut self, rhs: Float) {
        match &mut self.storage {
            MatrixStorage::Identity => {
                // Become diagonal with rhs on the main diagonal
                let n = self.n;
                self.data = vec![rhs; n];
                self.storage = MatrixStorage::Banded {
                    ml: 0,
                    mu: 0,
                };
            }
            MatrixStorage::Full => {
                for v in &mut self.data {
                    *v *= rhs;
                }
            }
            MatrixStorage::Banded { .. } => {
                for v in &mut self.data {
                    *v *= rhs;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix;

    #[test]
    fn mul_matrix_full() {
        let a: Matrix = Matrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let s = 5.0;
        let out = a.component_mul(s);
        assert_eq!(out[(0, 0)], 5.0);
        assert_eq!(out[(0, 1)], 10.0);
        assert_eq!(out[(1, 0)], 15.0);
        assert_eq!(out[(1, 1)], 20.0);
    }

    #[test]
    fn mul_identity() {
        let a: Matrix = Matrix::identity(2);
        let s = 5.0;
        let out = a.component_mul(s);
        assert_eq!(out[(0, 0)], 5.0);
        assert_eq!(out[(0, 1)], 0.0);
        assert_eq!(out[(1, 0)], 0.0);
        assert_eq!(out[(1, 1)], 5.0);
    }

    #[test]
    fn mul_assign() {
        let a: Matrix = Matrix::identity(2);
        let s = 5.0;
        let a = a.component_mul(s);
        assert_eq!(a[(0, 0)], 5.0);
        assert_eq!(a[(0, 1)], 0.0);
        assert_eq!(a[(1, 0)], 0.0);
        assert_eq!(a[(1, 1)], 5.0);
    }
}
