# Rust Examples

This file collects one canonical Rust example for each supported system shape:

- first-order systems with `FirstOrderSystem`
- second-order systems with `SecondOrderSystem`
- separable Hamiltonian systems with `SeparableHamiltonianSystem`

All three examples are available as runnable files under `examples/`.

## First-Order System

Use `Ivp::first_order(...)` for a generic system `y' = f(t, y)`.

Source: [`examples/first_order_decay.rs`](../examples/first_order_decay.rs)

```rust
use ivp::prelude::*;

struct ExponentialDecay {
    rate: f64,
}

impl FirstOrderSystem for ExponentialDecay {
    fn derivative(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = -self.rate * y[0];
    }
}

fn main() {
    let system = ExponentialDecay { rate: 0.5 };
    let y0 = [1.0];
    let t_eval: Vec<f64> = (0..=10).map(|i| i as f64).collect();

    let sol = Ivp::first_order(&system, 0.0, 10.0, &y0)
        .method(Method::DOPRI5)
        .rtol(1e-8)
        .atol(1e-10)
        .t_eval(t_eval)
        .solve()
        .unwrap();

    for (t, y) in sol.iter() {
        println!("t = {t:.1}, y = {:.7}", y[0]);
    }
}
```

Run it with:

```bash
cargo run --example first_order_decay
```

## Second-Order System

Use `Ivp::second_order(...)` when the problem has the restricted form
`q'' = a(t, q)`. This is the clean path for Verlet-style symplectic methods.

Source: [`examples/second_order_oscillator.rs`](../examples/second_order_oscillator.rs)

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

    println!("{:?}", sol.y.last().unwrap());
}
```

Run it with:

```bash
cargo run --example second_order_oscillator
```

## Hamiltonian System

Use `Ivp::hamiltonian(...)` for separable canonical systems with split dynamics
`q' = dT/dp(p)` and `p' = -dV/dq(q)`.

Source: [`examples/hamiltonian_oscillator.rs`](../examples/hamiltonian_oscillator.rs)

```rust
use ivp::prelude::*;

struct HarmonicHamiltonian;

impl SeparableHamiltonianSystem for HarmonicHamiltonian {
    fn position_derivative(&self, _t: f64, p: &[f64], dqdt: &mut [f64]) {
        dqdt[0] = p[0];
    }

    fn momentum_derivative(&self, _t: f64, q: &[f64], dpdt: &mut [f64]) {
        dpdt[0] = -q[0];
    }
}

fn main() {
    let q0 = [1.0];
    let p0 = [0.0];

    let sol = Ivp::hamiltonian(&HarmonicHamiltonian, 0.0, 20.0, &q0, &p0)
        .method(SymplecticMethod::Yoshida4)
        .step_size(0.05)
        .solve()
        .unwrap();

    println!("{:?}", sol.y.last().unwrap());
}
```

Run it with:

```bash
cargo run --example hamiltonian_oscillator
```
