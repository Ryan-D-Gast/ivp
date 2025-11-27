# Performance Improvement Suggestions for `ivp`

This document outlines performance optimizations and API improvements for the `ivp` crate, a Rust implementation of SciPy's `solve_ivp` with Python bindings.

---

## Critical Performance Improvements

### 1. Python Binding Overhead ✅ COMPLETED

**File:** `src/python.rs`

**Problem:** The Python ODE function is called through PyO3 on every evaluation, creating significant overhead by allocating new numpy arrays and tuples on each call.

```rust
// BEFORE: Creates new numpy arrays and tuples on every call
let y_arr = PyArray1::from_slice(self.py, y);
let args = PyTuple::new(self.py, ...);
let result = self.fun.call1(args)?;

// AFTER: Reuses pre-allocated numpy array, updates in-place
self.update_y_buffer(y);  // Uses unsafe ptr::copy_nonoverlapping
let args = self.build_call_args(x);
```

**Implemented:**
- [x] Pre-allocate numpy array in `PythonIVP::new()` 
- [x] Update numpy array in-place using unsafe `ptr::copy_nonoverlapping`
- [x] Add `RefCell` for lazy event buffer initialization
- [x] Add `#[inline(always)]` to hot path functions

**Remaining:**
- [ ] Implement actual `vectorized=true` mode that batches multiple evaluations

---

### 2. Memory Allocations in Hot Paths

**Files:** `src/methods/dopri5.rs`, `src/methods/dop853.rs`, `src/methods/radau.rs`, `src/methods/bdf.rs`

**Problem:** Workspace vectors are allocated on every `solve()` call.

```rust
// Current: Allocates workspace on every solve() call
let mut k1 = vec![0.0; n];
let mut k2 = vec![0.0; n];
// ... many more
```

**Suggestions:**
- [ ] Pre-allocate workspace buffers and reuse them across steps
- [ ] Consider adding a `Workspace<N>` struct that can be passed in or reused
- [ ] Use `SmallVec` from the `smallvec` crate for small state vectors (common case: n < 16)

---

### 3. SIMD/Vectorization Opportunities

**Files:** Stage computations in all solvers

**Problem:** Scalar operations in tight loops miss vectorization opportunities.

```rust
// Current: Scalar operations in tight loops
for i in 0..n {
    y1[i] = y[i] + h * (A31 * k1[i] + A32 * k2[i]);
}
```

**Suggestions:**
- [ ] Use explicit SIMD via `std::simd` (nightly) or `packed_simd2`/`wide` crates
- [ ] Alternatively, use `ndarray` with BLAS backend for large systems
- [ ] Consider using `rayon` for parallel stage evaluations when n is large (> 1000)
- [ ] Add `#[inline(always)]` to hot loop functions

---

### 4. Dense Output Collection Overhead

**File:** `src/solve/solout.rs`

**Problem:** Vectors are cloned on every accepted step for dense output.

```rust
// Current: Clones vectors on every step
let (cont, cxold, h) = interpolator.unwrap().get_cont();
// get_cont() does: cont_slice.to_vec()
```

**Suggestions:**
- [ ] Use `Cow<[Float]>` or store references with proper lifetime management
- [ ] Consider lazy evaluation - only materialize dense segments when accessed
- [ ] Add option to skip dense output storage for steps far from requested `t_eval` points

---

### 5. Event Detection Allocations

**File:** `src/solve/solout.rs`

**Problem:** Temporary buffers are allocated on every step when events are enabled.

```rust
// Current: Allocates on every step with events
let mut y_mid_buf = vec![0.0; y.len()];
let mut g_mid_vec = vec![0.0; n_events];
```

**Suggestions:**
- [ ] Pre-allocate these buffers in `DefaultSolOut::new()` and reuse
- [ ] Only allocate when crossing is detected, not on every step

---

### 6. LU Decomposition and Linear Algebra

**File:** `src/matrix/lu.rs`

**Problem:** Current implementation is straightforward but not optimized for performance.

**Suggestions:**
- [ ] Consider using `faer` or `nalgebra` with BLAS/LAPACK for large matrices
- [ ] Add feature flags for optional high-performance backends:
  ```toml
  [features]
  openblas = ["ndarray-linalg/openblas"]
  intel-mkl = ["ndarray-linalg/intel-mkl"]
  ```
- [ ] For small matrices (n < 8), consider inline unrolled implementations

---

### 7. Tolerance Indexing Overhead

**File:** `src/methods/mod.rs`

**Problem:** Match statement on every tolerance access in hot loops.

```rust
// Current: Branches on every access
impl Index<usize> for Tolerance {
    fn index(&self, index: usize) -> &Self::Output {
        match self {
            Tolerance::Scalar(v) => v,
            Tolerance::Vector(vs) => &vs[index],
        }
    }
}
```

**Suggestions:**
- [ ] Expand scalar tolerances to vectors at the start of `solve()` to eliminate per-access branching
- [ ] Or use a trait-based approach that monomorphizes

---

## Compiler and Build Optimizations

### 8. Cargo Profile Settings ✅ COMPLETED

**File:** `Cargo.toml`

**Implemented aggressive release optimizations:**

```toml
# Optimized release profile for maximum performance
[profile.release]
lto = "fat"
codegen-units = 1
opt-level = 3
strip = true

[profile.release.build-override]
opt-level = 3
```

---

### 9. Floating Point Type Flexibility

**File:** `Cargo.toml`, all source files

**Problem:** Currently uses feature flags for `f32`/`f64` but isn't truly generic.

**Suggestions:**
- [ ] Make core algorithms generic over `F: Float` trait
- [ ] This enables future `f128` support and allows users to choose at call-site
- [ ] Use `num-traits::Float` for trait bounds

---

## Python-Specific Optimizations

### 10. GIL and Zero-Copy

**File:** `src/python.rs`

**Suggestions:**
- [ ] Release the GIL during pure-Rust computation phases (not possible when calling Python callbacks, but possible for post-processing)
- [ ] Add a pure-Rust ODE specification option for Python users (e.g., via expression strings or Numba-compiled functions)
- [ ] Consider zero-copy for result arrays where possible

---

## API Compatibility with SciPy

### 11. Missing SciPy Parameters

**Suggestions:**
- [ ] Add `jac_sparsity` - sparse Jacobian patterns
- [ ] Expose `lband`/`uband` - banded Jacobian specification (exists internally)
- [ ] Add `min_step` to Python signature

---

### 12. Vectorized Mode Implementation

**Problem:** `vectorized=True` parameter is accepted but ignored.

**Suggestion:**
- [ ] Implement actual vectorized evaluation for when users set `vectorized=True`:
  ```python
  # Should support: fun(t, y) where y.shape = (n, k)
  ```

---

## Quick Wins (Low Effort, High Impact)

### 13. Inline Attributes

- [ ] Add `#[inline]` attributes to all small methods and closures
- [ ] Add `#[inline(always)]` to hot path functions

### 14. Memory Layout

- [ ] Use `Box<[Float]>` instead of `Vec<Float>` for fixed-size buffers (avoids capacity tracking)

### 15. Branch Prediction Hints

- [ ] Add `#[cold]` to error paths to improve branch prediction

### 16. Slice Operations

- [ ] Use `copy_from_slice` instead of element-wise loops where applicable (partially done)

---

## Benchmarking Infrastructure

### 17. Expanded Benchmarks

**Suggestions:**
- [ ] Add memory allocation profiling with `dhat` or `heaptrack`
- [ ] Add Criterion.rs benchmarks for Rust-only testing
- [ ] Add profile-guided optimization (PGO) support
- [ ] Compare against Julia's DifferentialEquations.jl for additional reference

---

## Current Performance (Baseline After Optimizations #1, #8)

Benchmark results comparing `ivp` (Rust) vs `scipy.integrate.solve_ivp`:

| Benchmark | IVP Time | SciPy Time | Speedup |
|-----------|----------|------------|---------|
| Van der Pol (DOP853, non-stiff) | 0.015s | 0.057s | **3.8x** |
| Van der Pol (RK45, non-stiff) | 0.013s | 0.071s | **5.7x** |
| Van der Pol (BDF, stiff) | 0.013s | 0.186s | **14.0x** |
| Linear System (N=1000, RK45) | 0.0007s | 0.003s | **3.8x** |

*Tested on Snapdragon X Elite, x64 Python, Windows 11*

---

## Implementation Priority

| Priority | Item | Impact | Effort | Status |
|----------|------|--------|--------|--------|
| 🔴 High | Python binding overhead (#1) | Very High | Medium | ✅ Done |
| 🔴 High | Memory allocations in hot paths (#2) | High | Medium | |
| 🔴 High | Cargo profile settings (#8) | Medium | Low | ✅ Done |
| 🟡 Medium | Event detection allocations (#5) | Medium | Low | |
| 🟡 Medium | Dense output overhead (#4) | Medium | Medium | |
| 🟡 Medium | Quick wins (#13-16) | Medium | Low | |
| 🟢 Low | SIMD vectorization (#3) | High | High | |
| 🟢 Low | LU decomposition backends (#6) | High | High | |
| 🟢 Low | Generic float types (#9) | Low | High | |

---

## Notes

- All changes should maintain backward compatibility with existing API
- Performance improvements should be validated with benchmarks before/after
- Consider adding feature flags for optional optimizations that increase compile time
