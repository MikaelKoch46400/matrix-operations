# matrix-operations

Fixed-size matrix multiplication, transpose, and determinant for small square matrices.

```rust
use matrix_operations::Mat;

let a = Mat::from_row_major(vec![1.0, 2.0, 3.0, 4.0]);
let b = Mat::identity(2);

let c = a.mul(&b).unwrap();
assert_eq!(c, a);

let t = a.transpose();
assert_eq!(t.get(0, 1), 3.0);

let d = a.determinant().unwrap(); // -2.0
```

## Why this exists

Most matrix crates are general-purpose: they handle rectangular matrices,
arbitrary dimensions, decompositions, and more. That generality costs code
size and compile time. This library is deliberately narrow. It targets the
case where you have a handful of small square matrices (2x2 through 4x4)
and you want a handful of operations without pulling in a dependency.

The trade-off is that the determinant is computed by cofactor expansion,
which is O(n!) and only practical for very small `n`. Rather than let a
caller accidentally ask for the determinant of a 20x20 matrix and wait
forever, `determinant` returns `None` for `n >= 5`. If you need larger
matrices, this is not the library for you.

## API

All types and functions live in the crate root.

- `Mat<T>` — a square matrix stored in row-major order.
  - `Mat::filled(n, value)` — `n x n` matrix filled with `value`.
  - `Mat::from_row_major(data)` — build from a flat row-major `Vec`. Panics if the length is not a perfect square.
  - `Mat::identity(n)` — `n x n` identity (only for `Mat<f64>`).
  - `dim()` — the dimension `n`.
  - `get(r, c)` — element access. Panics on out-of-bounds.
  - `set(r, c, value)` — element assignment. Panics on out-of-bounds.
  - `transpose()` — returns the transpose.
  - `mul(&rhs)` — returns `Option<Mat<f64>>`; `None` on dimension mismatch.
  - `determinant()` — returns `Option<f64>`; `None` for `n >= 5`.

`Mat<T>` implements `Clone`, `Debug`, `PartialEq`, and `Display` (when `T: Display`).

## The awkward edge

The determinant is the awkward part. It is defined for every square matrix,
but computing it exactly by cofactor expansion is only feasible for small `n`.
This library chooses to return `None` above `n = 4` rather than silently
switch to a different algorithm or produce an approximation. The tests
exercise that boundary directly.
