//! Core matrix type, storage enum, and constructors.

use crate::Float;

/// Matrix storage layout.
#[derive(PartialEq, Clone, Debug)]
pub enum MatrixStorage {
    /// Identity matrix (implicit). `data` stores [one, zero] to satisfy indexing by reference.
    Identity,
    /// Dense row-major matrix (nrows*ncols entries).
    Full,
    /// Banded matrix with lower (ml) and upper (mu) bandwidth.
    /// Compact diagonal storage with shape (ml+mu+1, ncols), row-major per diagonal.
    /// Off-band reads return a shared constant zero.
    Banded { ml: usize, mu: usize },
}

/// Generic matrix for linear algebra (typically square in current use).
#[derive(PartialEq, Clone, Debug)]
pub struct Matrix {
    pub n: usize,
    pub m: usize,
    pub data: Vec<Float>,
    pub storage: MatrixStorage,
}

impl Matrix {
    /// Number of rows.
    pub fn nrows(&self) -> usize {
        self.n
    }

    /// Number of columns.
    pub fn ncols(&self) -> usize {
        self.m
    }

    /// Identity matrix of size n x n.
    pub fn identity(n: usize) -> Self {
        Matrix {
            n,
            m: n,
            // Keep [one, zero] so indexing can return references.
            data: vec![1.0, 0.0],
            storage: MatrixStorage::Identity,
        }
    }

    /// Creates a matrix from a vector.
    pub fn from_vec(n: usize, m: usize, data: Vec<Float>) -> Self {
        assert_eq!(data.len(), n * m, "Incompatible data length");
        Matrix {
            n,
            m,
            data,
            storage: MatrixStorage::Full,
        }
    }

    /// Creates a matrix from a storage type.
    pub fn from_storage(n: usize, m: usize, storage: MatrixStorage) -> Self {
        let data = match storage {
            MatrixStorage::Identity => vec![1.0, 0.0],
            MatrixStorage::Full => vec![0.0; n * m],
            MatrixStorage::Banded { ml, mu } => vec![0.0; (ml + mu + 1) * n],
        };
        Matrix {
            n,
            m,
            data,
            storage,
        }
    }

    /// Full matrix from a row-major vector of length n*m.
    pub fn full(n: usize, m: usize) -> Self {
        let data = vec![0.0; n * m];
        Matrix {
            n,
            m,
            data,
            storage: MatrixStorage::Full,
        }
    }

    /// Square matrix of size n x n.
    pub fn square(n: usize) -> Self {
        Matrix {
            n,
            m: n,
            data: Vec::with_capacity(n * n),
            storage: MatrixStorage::Full,
        }
    }

    /// Zero matrix of size n x m.
    pub fn zeros(n: usize, m: usize) -> Self {
        Matrix {
            n,
            m,
            data: vec![0.0; n * m],
            storage: MatrixStorage::Full,
        }
    }

    /// Zero banded matrix with the given bandwidths.
    /// For entry (i,j) within the band, index maps to data[i - j + mu, j].
    pub fn banded(n: usize, ml: usize, mu: usize) -> Self {
        let rows = ml + mu + 1;
        let data = vec![0.0; rows * n];
        Matrix {
            n,
            m: n,
            data,
            storage: MatrixStorage::Banded { ml, mu },
        }
    }

    /// Diagonal matrix from the provided diagonal entries (ml=mu=0).
    pub fn diagonal(diag: Vec<Float>) -> Self {
        let n = diag.len();
        // With ml=mu=0, storage is (1,n), so `diag` maps directly to row 0.
        Matrix {
            n,
            m: n,
            data: diag,
            storage: MatrixStorage::Banded { ml: 0, mu: 0 },
        }
    }

    /// Zero lower-triangular matrix (ml = n-1, mu = 0).
    pub fn lower_triangular(n: usize) -> Self {
        Matrix::banded(n, n.saturating_sub(1), 0)
    }

    /// Zero upper-triangular matrix (ml = 0, mu = n-1).
    pub fn upper_triangular(n: usize) -> Self {
        Matrix::banded(n, 0, n.saturating_sub(1))
    }

    /// Dimensions (nrows, ncols).
    pub fn dims(&self) -> (usize, usize) {
        (self.n, self.m)
    }

    /// Checks if the matrix is an identity matrix.
    pub fn is_identity(&self) -> bool {
        match self.storage {
            MatrixStorage::Identity => true,
            _ => {
                for i in 0..self.n {
                    for j in 0..self.m {
                        let val = self[(i, j)];
                        if i == j {
                            if val != 1.0 {
                                return false;
                            }
                        } else if val != 0.0 {
                            return false;
                        }
                    }
                }
                true
            }
        }
    }

    /// Swap two rows in-place for Full storage. For Banded storage, performs a logical swap
    /// of accessible entries within the band; for Identity, no-op unless swapping equal indices.
    pub fn swap_rows(&mut self, r1: usize, r2: usize) {
        assert!(r1 < self.n && r2 < self.n, "row index out of bounds");
        if r1 == r2 {
            return;
        }
        match &mut self.storage {
            MatrixStorage::Full => {
                for j in 0..self.m {
                    self.data.swap(r1 * self.m + j, r2 * self.m + j);
                }
            }
            MatrixStorage::Identity => {
                // Identity is stored as [one, zero]; swapping has no effect on implicit structure.
                // Clients should not attempt to permute Identity rows; we ignore to keep API simple.
            }
            MatrixStorage::Banded { ml, mu, .. } => {
                // Only swap entries that are actually stored (within band).
                // For each column j, if (r1,j) and/or (r2,j) are in band, swap.
                let mlv = *ml as isize;
                let muv = *mu as isize;
                for j in 0..self.m {
                    let k1 = r1 as isize - j as isize;
                    let k2 = r2 as isize - j as isize;
                    let in1 = k1 >= -muv && k1 <= mlv;
                    let in2 = k2 >= -muv && k2 <= mlv;
                    if in1 && in2 {
                        let row1 = (k1 + *mu as isize) as usize;
                        let row2 = (k2 + *mu as isize) as usize;
                        self.data.swap(row1 * self.m + j, row2 * self.m + j);
                    } else if in1 || in2 {
                        // One entry is implicit zero; swapping sets stored one to zero and vice versa
                        // This best-effort maintains logical swap within band footprint.
                        if in1 {
                            let row1 = (k1 + *mu as isize) as usize;
                            let idx1 = row1 * self.m + j;
                            self.data[idx1] = 0.0;
                        } else {
                            let row2 = (k2 + *mu as isize) as usize;
                            let idx2 = row2 * self.m + j;
                            self.data[idx2] = 0.0;
                        }
                    }
                }
            }
        }
    }

    /// Fill the matrix with a constant value.
    pub fn fill(&mut self, value: Float) {
        self.data.fill(value);
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix;

    #[test]
    fn diagonal_constructor_sets_diagonal() {
        let m = Matrix::diagonal(vec![1.0f64, 2.0, 3.0]);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(1, 1)], 2.0);
        assert_eq!(m[(2, 2)], 3.0);
        assert_eq!(m[(0, 1)], 0.0);
        assert_eq!(m[(2, 0)], 0.0);
    }

    #[test]
    fn triangular_constructors_shape() {
        let l: Matrix = Matrix::lower_triangular(4);
        // Above main diagonal reads zero
        assert_eq!(l[(0, 3)], 0.0);
        let u: Matrix = Matrix::upper_triangular(4);
        // Below main diagonal reads zero
        assert_eq!(u[(3, 0)], 0.0);
    }
}
