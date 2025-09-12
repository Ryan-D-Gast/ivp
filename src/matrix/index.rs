//! Indexing and display for `Matrix`.

use core::fmt::{self, Display, Write as _};
use core::ops::{Index, IndexMut};

use crate::Float;

use super::base::{Matrix, MatrixStorage};

/// 2D indexing by (i, j), read-only.
impl Index<(usize, usize)> for Matrix {
    type Output = Float;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (i, j) = index;
        assert!(i < self.n && j < self.m, "Index out of bounds");
        match &self.storage {
            MatrixStorage::Identity => {
                if i == j {
                    &self.data[0]
                } else {
                    &self.data[1]
                }
            }
            MatrixStorage::Full => &self.data[i * self.m + j],
            MatrixStorage::Banded { ml, mu, zero } => {
                let k = i as isize - j as isize;
                if k < -(*mu as isize) || k > *ml as isize {
                    zero
                } else {
                    let row = (k + *mu as isize) as usize;
                    &self.data[row * self.m + j]
                }
            }
        }
    }
}

/// 2D indexing by (i, j), mutable (where supported).
impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        assert!(i < self.n && j < self.m, "Index out of bounds");
        match &mut self.storage {
            MatrixStorage::Full => &mut self.data[i * self.m + j],
            MatrixStorage::Identity => {
                panic!(
                    "cannot mutate Identity matrix via indexing; convert explicitly to Full first"
                )
            }
            MatrixStorage::Banded { ml, mu, .. } => {
                let k = i as isize - j as isize;
                if k >= -(*mu as isize) && k <= *ml as isize {
                    let row = (k + *mu as isize) as usize;
                    &mut self.data[row * self.m + j]
                } else {
                    panic!(
                        "attempted to write outside band of Banded matrix: i-j={} not in [-mu, ml] = [-{}, {}]",
                        k, mu, ml
                    )
                }
            }
        }
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (nr, nc) = (self.n, self.m);
        for i in 0..nr {
            f.write_str("[")?;
            for j in 0..nc {
                if j > 0 {
                    f.write_str(" ")?;
                }
                write!(f, "{}", self[(i, j)])?;
            }
            f.write_str("]")?;
            if i + 1 < nr {
                f.write_char('\n')?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;

    #[test]
    #[should_panic(expected = "cannot mutate Identity matrix via indexing")]
    fn identity_panics_on_write() {
        let mut m: Matrix = Matrix::identity(3);
        m[(2, 0)] = 5.0; // should panic
    }

    #[test]
    fn banded_inband_write_keeps_banded() {
        let mut b: Matrix = Matrix::banded(5, 1, 2);
        b[(2, 2)] = 10.0; // main diag
        b[(1, 2)] = 5.0; // upper-1
        b[(3, 2)] = 7.0; // lower-1
        // stays valid and values accessible
        assert_eq!(b[(2, 2)], 10.0);
        assert_eq!(b[(1, 2)], 5.0);
        assert_eq!(b[(3, 2)], 7.0);
        // Outside band reads as zero
        assert_eq!(b[(0, 4)], 0.0);
    }

    #[test]
    #[should_panic(expected = "attempted to write outside band of Banded matrix")]
    fn banded_out_of_band_write_panics() {
        let mut b: Matrix = Matrix::banded(3, 0, 0); // only main diagonal
        b[(0, 2)] = 7.0; // out-of-band -> panic
    }

    #[test]
    fn full_index_read_write() {
        let mut m: Matrix = Matrix::zeros(2, 2);
        m[(0, 0)] = 1.0;
        m[(0, 1)] = 2.0;
        m[(1, 0)] = 3.0;
        m[(1, 1)] = 4.0;
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(1, 0)], 3.0);
    }

    #[test]
    fn display_prints_dense_matrix() {
        let m: Matrix = Matrix::identity(3);
        let s = format!("{}", m);
        assert!(s.contains("[1 0 0]"));
        assert!(s.contains("[0 1 0]"));
        assert!(s.contains("[0 0 1]"));
    }
}
