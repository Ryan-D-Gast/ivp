<p align="center">
  <img src="./assets/logo.svg" width="1000" alt="ivp">
</p>

<p align="center">
    <a href="https://crates.io/crates/ivp">
        <img src="https://img.shields.io/crates/v/ivp.svg?style=flat-square" alt="crates.io">
    </a>
    <a href="https://pypi.org/project/ivp-rs/">
        <img src="https://img.shields.io/pypi/v/ivp-rs.svg?style=flat-square" alt="PyPI">
    </a>
    <a href="https://docs.rs/ivp">
        <img src="https://docs.rs/ivp/badge.svg" alt="docs.rs">
    </a>
    <a href="https://github.com/Ryan-D-Gast/ivp/blob/main/LICENSE">
        <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
    </a>
</p>

<p align="center">
    <strong>
        <a href="https://docs.rs/ivp/latest/ivp/">Documentation</a> |
        <a href="./examples/">Examples</a> |
        <a href="https://github.com/Ryan-D-Gast/ivp">GitHub</a> |
        <a href="https://crates.io/crates/ivp">Crates.io</a> |
        <a href="https://pypi.org/project/ivp-rs/">PyPI</a>
    </strong>
</p>

-----

<p align="center">
<b>A library of numerical methods for solving initial value problems (IVPs)</b><br>
<i>for Rust and Python.</i>
</p>

-----

This library provides a pure Rust implementation of SciPy's `solve_ivp` functionality with a typed builder API for first-order, second-order, and Hamiltonian systems. It is also available as a Python package with a SciPy-compatible `solve_ivp` interface.

## Features

Currently implemented solvers:
-   **DOP853**: An 8th order Dormand-Prince method with step-size control and dense output.
-   **DOPRI5**: A 5th order Dormand-Prince method with step-size control and dense output.
-   **RK4**: The classic 4th order Runge-Kutta method with fixed step-size and cubic Hermite interpolation for dense output.
-   **RK23**: A 3rd order Runge-Kutta method with 2nd order error estimate for step-size control.
-   **LSODA**: An automatic Adams/BDF switching multistep method for problems that may change stiffness.
-   **Radau**: A 5th order implicit Runge-Kutta method of Radau IIA type with step-size control and dense output.
-   **BDF**: A variable-order (1 to 5) Backward Differentiation Formula method for stiff ODEs with adaptive step-size control and dense output.
-   **Symplectic methods**: Fixed-step structured solvers for separable Hamiltonian and second-order systems, including Symplectic Euler, Velocity Verlet, Ruth 3, and Yoshida 4.

## Installation

### Rust

```bash
cargo add ivp
```

### Python

```bash
pip install ivp-rs
```

## Example Usage

```python
from ivp import solve_ivp

def exponential_decay(t, y):
    return -0.5 * y

# Solve the ODE
sol = solve_ivp(exponential_decay, (0, 10), [1.0], method='RK45', rtol=1e-6, atol=1e-9)

print(f"Final time: {sol.t[-1]}")
print(f"Final state: {sol.y[:, -1]}")
```

```rust
use ivp::prelude::*;

struct ExponentialDecay;

impl FirstOrderSystem for ExponentialDecay {
    fn derivative(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = -0.5 * y[0];
    }
}

fn main() {
    let decay = ExponentialDecay;
    let y0 = [1.0];

    let sol = Ivp::first_order(&decay, 0.0, 10.0, &y0)
        .method(Method::DOPRI5)
        .rtol(1e-6)
        .atol(1e-9)
        .solve()
        .unwrap();
    
    println!("Final time: {}", sol.t.last().unwrap());
    println!("Final state: {:?}", sol.y.last().unwrap());
}
```

More complete examples:
- Rust: [`docs/rust-examples.md`](./docs/rust-examples.md)
- Python: [`docs/python-examples.md`](./docs/python-examples.md)

## Symplectic Usage

For structured mechanics problems, use the dedicated second-order or Hamiltonian
builders instead of the Rust first-order `Ivp::first_order(...)` path.

```rust
use ivp::prelude::*;

struct HarmonicOscillator;

impl SecondOrderSystem for HarmonicOscillator {
    fn acceleration(&self, _t: f64, q: &[f64], a: &mut [f64]) {
        a[0] = -q[0];
    }
}

fn main() {
    let q0 = [1.0];
    let v0 = [0.0];

    let sol = Ivp::second_order(&HarmonicOscillator, 0.0, 20.0, &q0, &v0)
        .method(SymplecticMethod::VelocityVerlet)
        .step_size(0.05)
        .solve()
        .unwrap();

    println!("Final state: {:?}", sol.y.last().unwrap());
}
```

Python can also route these methods through `solve_ivp`. Second-order problems
can use a plain acceleration callback, and Hamiltonian problems can use a
callback pair:

```python
import numpy as np
from ivp import solve_ivp

def acceleration(t, q):
    return -np.asarray(q, dtype=float)

sol = solve_ivp(
    acceleration,
    (0.0, 20.0),
    [1.0, 0.0, -0.5, 0.0, 1.0, 0.25],  # [q..., v...]
    method="VelocityVerlet",
    step_size=0.05,
    dense_output=True,
)

print(sol.sol(0.5))
```

```python
def position_derivative(t, p):
    return np.array([p[0]])

def momentum_derivative(t, q):
    return np.array([-q[0]])

sol = solve_ivp(
    (position_derivative, momentum_derivative),
    (0.0, 20.0),
    [1.0, 0.0],
    method="Yoshida4",
    step_size=0.05,
)
```

On the Python side, invalid callback shapes now raise normal Python exceptions
with explicit messages. For example, a symplectic callback returning the wrong
number of values raises `ValueError`, and an incomplete Hamiltonian callback
pair raises `TypeError`.

Current limitation: event handling is not yet supported for symplectic methods.
