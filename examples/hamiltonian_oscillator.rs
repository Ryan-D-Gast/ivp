//! Hamiltonian example: harmonic oscillator in canonical coordinates.

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

fn energy(state: &[f64]) -> f64 {
    let q = state[0];
    let p = state[1];
    0.5 * (q * q + p * p)
}

fn main() {
    let q0 = [1.0];
    let p0 = [0.0];
    let t_eval: Vec<f64> = (0..=10).map(|i| 2.0 * i as f64).collect();

    let sol = Ivp::hamiltonian(&HarmonicHamiltonian, 0.0, 20.0, &q0, &p0)
        .method(SymplecticMethod::Yoshida4)
        .step_size(0.05)
        .t_eval(t_eval)
        .solve()
        .unwrap();

    let h0 = energy(&sol.y[0]);
    println!("status: {:?}", sol.status);
    for (t, y) in sol.iter() {
        println!(
            "t = {:>5.2}, q = {:>9.6}, p = {:>9.6}, dH = {:>9.2e}",
            t,
            y[0],
            y[1],
            energy(y) - h0
        );
    }
}
