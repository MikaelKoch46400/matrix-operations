//! Fixed-size matrix operations for small square matrices.
//!
//! The matrices are stored in row-major order as a flat `Vec<f64>`.
//! All operations are implemented for arbitrary square dimensions, but the
//! intended use is small matrices (2x2, 3x3, 4x4) where the constant factors
//! of a general implementation do not matter.
//!
//! # Design decisions
//!
//! * `Mat<T>` is generic over the element type. The arithmetic operations
//!   are only implemented for `f64` because that is what the problem domain
//!   needs, and keeping the trait bounds narrow avoids pulling in numeric
//!   traits from external crates.
//! * The determinant uses cofactor expansion along the first row. This is
//!   O(n!) and therefore only suitable for small `n`. For `n >= 5` we return
//!   `None` rather than attempting an expensive computation. This is a
//!   deliberate trade-off: the library is for small matrices, and a silent
//!   factorial blow-up is worse than an explicit refusal.
//! * Multiplication returns `None` when the dimensions do not match. The
//!   alternative (panicking) would make the function unusable in contexts
//!   where the dimensions come from user input.

use std::fmt;

/// A dense square matrix stored in row-major order.
///
/// The element at row `r` and column `c` lives at `data[r * n + c]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Mat<T> {
    n: usize,
    data: Vec<T>,
}

impl<T: Copy> Mat<T> {
    /// Creates a square matrix of size `n` filled with `value`.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`. A zero-dimensional matrix has no meaningful
    /// operations and is almost always a bug in the caller.
    pub fn filled(n: usize, value: T) -> Self {
        assert!(n > 0, "matrix dimension must be non-zero");
        Mat {
            n,
            data: vec![value; n * n],
        }
    }

    /// Creates a matrix from a flat row-major vector.
    ///
    /// # Panics
    ///
    /// Panics if `data.len()` is not a perfect square, or if it is zero.
    pub fn from_row_major(data: Vec<T>) -> Self {
        let len = data.len();
        assert!(len > 0, "matrix must have at least one element");
        let n = (len as f64).sqrt() as usize;
        assert!(n * n == len, "data length must be a perfect square");
        Mat { n, data }
    }

    /// Returns the dimension `n` of the `n x n` matrix.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Returns the element at row `r`, column `c`.
    ///
    /// # Panics
    ///
    /// Panics if `r >= n` or `c >= n`.
    pub fn get(&self, r: usize, c: usize) -> T {
        assert!(r < self.n && c < self.n, "index out of bounds");
        self.data[r * self.n + c]
    }

    /// Sets the element at row `r`, column `c`.
    ///
    /// # Panics
    ///
    /// Panics if `r >= n` or `c >= n`.
    pub fn set(&mut self, r: usize, c: usize, value: T) {
        assert!(r < self.n && c < self.n, "index out of bounds");
        self.data[r * self.n + c] = value;
    }

    /// Returns the transpose of the matrix.
    ///
    /// The transpose swaps rows and columns: `(A^T)[i][j] = A[j][i]`.
    /// For a square matrix the result has the same dimension.
    pub fn transpose(&self) -> Self {
        let mut out = Mat {
            n: self.n,
            data: vec![self.data[0]; self.n * self.n],
        };
        for r in 0..self.n {
            for c in 0..self.n {
                out.data[c * self.n + r] = self.data[r * self.n + c];
            }
        }
        out
    }
}

impl Mat<f64> {
    /// Creates the `n x n` identity matrix.
    pub fn identity(n: usize) -> Self {
        let mut m = Mat::filled(n, 0.0);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    /// Multiplies two matrices.
    ///
    /// Returns `None` if the dimensions do not match. Because the matrices
    /// are square, the only mismatch possible is `self.n != rhs.n`.
    pub fn mul(&self, rhs: &Mat<f64>) -> Option<Mat<f64>> {
        if self.n != rhs.n {
            return None;
        }
        let n = self.n;
        let mut out = Mat::filled(n, 0.0);
        for i in 0..n {
            for k in 0..n {
                let a = self.data[i * n + k];
                for j in 0..n {
                    out.data[i * n + j] += a * rhs.data[k * n + j];
                }
            }
        }
        Some(out)
    }

    /// Computes the determinant by cofactor expansion along the first row.
    ///
    /// Returns `None` for `n >= 5` because the cost grows factorially and
    /// this library is only intended for small matrices. Returning `None`
    /// is preferable to a silent stall on a larger input.
    pub fn determinant(&self) -> Option<f64> {
        if self.n >= 5 {
            return None;
        }
        Some(det_recursive(&self.data, self.n))
    }
}

fn det_recursive(data: &[f64], n: usize) -> f64 {
    if n == 1 {
        return data[0];
    }
    if n == 2 {
        return data[0] * data[3] - data[1] * data[2];
    }
    let mut det = 0.0;
    let mut sign = 1.0;
    for col in 0..n {
        // Build the minor by removing row 0 and column `col`.
        let mut minor = Vec::with_capacity((n - 1) * (n - 1));
        for r in 1..n {
            for c in 0..n {
                if c != col {
                    minor.push(data[r * n + c]);
                }
            }
        }
        det += sign * data[col] * det_recursive(&minor, n - 1);
        sign = -sign;
    }
    det
}

impl<T: fmt::Display> fmt::Display for Mat<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for r in 0..self.n {
            write!(f, "[")?;
            for c in 0..self.n {
                if c > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.data[r * self.n + c])?;
            }
            writeln!(f, "]")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn assert_close(a: f64, b: f64) {
        assert!(
            (a - b).abs() < EPS,
            "expected {} to be close to {}",
            a,
            b
        );
    }

    #[test]
    fn identity_is_neutral_for_multiplication() {
        let a = Mat::from_row_major(vec![1.0, 2.0, 3.0, 4.0]);
        let i = Mat::identity(2);
        let left = a.mul(&i).unwrap();
        let right = i.mul(&a).unwrap();
        assert_eq!(left, a);
        assert_eq!(right, a);
    }

    #[test]
    fn multiplication_2x2() {
        let a = Mat::from_row_major(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Mat::from_row_major(vec![5.0, 6.0, 7.0, 8.0]);
        let c = a.mul(&b).unwrap();
        assert_close(c.get(0, 0), 19.0);
        assert_close(c.get(0, 1), 22.0);
        assert_close(c.get(1, 0), 43.0);
        assert_close(c.get(1, 1), 50.0);
    }

    #[test]
    fn multiplication_dimension_mismatch_returns_none() {
        let a = Mat::identity(2);
        let b = Mat::identity(3);
        assert!(a.mul(&b).is_none());
    }

    #[test]
    fn transpose_2x2() {
        let a = Mat::from_row_major(vec![1.0, 2.0, 3.0, 4.0]);
        let t = a.transpose();
        assert_close(t.get(0, 0), 1.0);
        assert_close(t.get(0, 1), 3.0);
        assert_close(t.get(1, 0), 2.0);
        assert_close(t.get(1, 1), 4.0);
    }

    #[test]
    fn transpose_is_involutive() {
        let a = Mat::from_row_major(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(a.transpose().transpose(), a);
    }

    #[test]
    fn determinant_1x1() {
        let a = Mat::from_row_major(vec![7.0]);
        assert_close(a.determinant().unwrap(), 7.0);
    }

    #[test]
    fn determinant_2x2() {
        let a = Mat::from_row_major(vec![1.0, 2.0, 3.0, 4.0]);
        assert_close(a.determinant().unwrap(), -2.0);
    }

    #[test]
    fn determinant_3x3() {
        let a = Mat::from_row_major(vec![
            6.0, 1.0, 1.0,
            4.0, -2.0, 5.0,
            2.0, 8.0, 7.0,
        ]);
        assert_close(a.determinant().unwrap(), -306.0);
    }

    #[test]
    fn determinant_singular_is_zero() {
        let a = Mat::from_row_major(vec![1.0, 2.0, 2.0, 4.0]);
        assert_close(a.determinant().unwrap(), 0.0);
    }

    #[test]
    fn determinant_4x4() {
        // Upper triangular matrix: determinant is the product of the diagonal.
        let a = Mat::from_row_major(vec![
            1.0, 2.0, 3.0, 4.0,
            0.0, 5.0, 6.0, 7.0,
            0.0, 0.0, 8.0, 9.0,
            0.0, 0.0, 0.0, 10.0,
        ]);
        assert_close(a.determinant().unwrap(), 400.0);
    }

    #[test]
    fn determinant_5x5_is_none() {
        let a = Mat::identity(5);
        assert!(a.determinant().is_none());
    }

    #[test]
    fn determinant_of_identity_is_one() {
        for n in 1..=4 {
            let a = Mat::identity(n);
            assert_close(a.determinant().unwrap(), 1.0);
        }
    }

    #[test]
    #[should_panic(expected = "matrix dimension must be non-zero")]
    fn filled_zero_panics() {
        let _ = Mat::filled(0, 0.0);
    }

    #[test]
    #[should_panic(expected = "data length must be a perfect square")]
    fn from_row_major_non_square_panics() {
        let _ = Mat::from_row_major(vec![1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn get_out_of_bounds_panics() {
        let a = Mat::identity(2);
        let _ = a.get(2, 0);
    }
}
