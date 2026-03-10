# Python Examples

This file collects one canonical Python example for each supported system shape:

- first-order systems through a plain `fun(t, y)` callback
- second-order systems through a plain `fun(t, q)` acceleration callback
- separable Hamiltonian systems through either a callback pair
  `(position_derivative, momentum_derivative)` or explicit keyword callbacks

All three examples are available as runnable files under `examples/`.

## Callback Rules

The Python `solve_ivp` entrypoint accepts different callback shapes depending on
the method family:

- standard first-order methods expect `fun(t, y)` returning a 1D array-like object with the same length as `y`
- symplectic second-order methods expect `fun(t, q)` or `acceleration=...`, returning length `len(y0) // 2`
- symplectic Hamiltonian methods expect either `(position_derivative, momentum_derivative)` or the equivalent keyword callbacks, each returning length `len(y0) // 2`

Common misuse now raises normal Python exceptions instead of exposing a Rust
panic. Typical mistakes are wrong output length, non-numeric return values,
odd-length symplectic initial states, or providing only one Hamiltonian
callback.

## First-Order System

Use `solve_ivp(...)` with a standard callable and one of the standard method
names such as `RK45`, `RK23`, `DOP853`, `Radau`, or `BDF`.

Source: [`examples/first_order_decay.py`](../examples/first_order_decay.py)

```python
import ivp
import numpy as np


def decay(t, y):
    return np.array([-0.5 * y[0]], dtype=float)


sol = ivp.solve_ivp(
    decay,
    (0.0, 10.0),
    [1.0],
    method="RK45",
    t_eval=np.linspace(0.0, 10.0, 11),
    rtol=1e-8,
    atol=1e-10,
)

print(sol.y[:, -1])
```

Run it with:

```bash
python examples/first_order_decay.py
```

## Second-Order System

For symplectic second-order integration, pass a callable
`fun(t, q)` returning the acceleration. In Python, the initial state is still flattened as
`[q..., v...]`.

For example, a 3D problem uses:

- `q = [x, y, z]`
- `v = [vx, vy, vz]`
- `y0 = [x, y, z, vx, vy, vz]`

The callback still only receives the position block `q`, not the flattened full
state.

Source: [`examples/second_order_oscillator.py`](../examples/second_order_oscillator.py)

```python
import ivp
import numpy as np


def acceleration(t, q):
    return -np.asarray(q, dtype=float)


y0 = [
    1.0, 0.0, -0.5,   # q0 = [x0, y0, z0]
    0.0, 1.0, 0.25,   # v0 = [vx0, vy0, vz0]
]

sol = ivp.solve_ivp(
    acceleration,
    (0.0, 20.0),
    y0,
    method="VelocityVerlet",
    step_size=0.05,
    t_eval=np.linspace(0.0, 20.0, 11),
)

print(sol.y[:, -1])
```

The returned solution uses the same layout, so for a 3D system:

- `sol.y[0:3, i]` is the position vector at output index `i`
- `sol.y[3:6, i]` is the velocity vector at output index `i`

Run it with:

```bash
python examples/second_order_oscillator.py
```

## Hamiltonian System

For symplectic Hamiltonian integration, pass a callback pair
`(position_derivative, momentum_derivative)` or use the equivalent keyword
arguments `position_derivative=...` and `momentum_derivative=...`. In Python,
the initial state is flattened as `[q..., p...]`.

Source: [`examples/hamiltonian_oscillator.py`](../examples/hamiltonian_oscillator.py)

```python
import ivp
import numpy as np


def position_derivative(t, p):
    return np.array([p[0]], dtype=float)


def momentum_derivative(t, q):
    return np.array([-q[0]], dtype=float)


sol = ivp.solve_ivp(
    (position_derivative, momentum_derivative),
    (0.0, 20.0),
    [1.0, 0.0],
    method="Yoshida4",
    step_size=0.05,
    t_eval=np.linspace(0.0, 20.0, 11),
)

print(sol.y[:, -1])
```

Run it with:

```bash
python examples/hamiltonian_oscillator.py
```

You can also write the Hamiltonian call in keyword form:

```python
sol = ivp.solve_ivp(
    None,
    (0.0, 20.0),
    [1.0, 0.0],
    method="Yoshida4",
    step_size=0.05,
    position_derivative=position_derivative,
    momentum_derivative=momentum_derivative,
)
```
