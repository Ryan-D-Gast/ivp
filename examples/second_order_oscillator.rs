//! Second-order example: harmonic oscillator with Velocity Verlet.

use ivp::prelude::*;

struct HarmonicOscillator;

impl SecondOrderSystem for HarmonicOscillator {
    fn acceleration(&self, _t: f64, q: &[f64], _p: &[f64], a: &mut [f64]) {
        a[0] = -q[0];
    }
}

fn energy(state: &[f64]) -> f64 {
    let q = state[0];
    let v = state[1];
    0.5 * (q * q + v * v)
}

fn main() {
    let q0 = [1.0];
    let v0 = [0.0];
    let t_eval: Vec<f64> = (0..=10).map(|i| 2.0 * i as f64).collect();

    let sol = Ivp::second_order(&HarmonicOscillator, 0.0, 20.0, &q0, &v0)
        .method(SymplecticMethod::VelocityVerlet)
        .step_size(0.05)
        .t_eval(t_eval)
        .solve()
        .unwrap();

    let e0 = energy(&sol.y[0]);
    println!("status: {:?}", sol.status);
    for (t, y) in sol.iter() {
        println!(
            "t = {:>5.2}, q = {:>9.6}, v = {:>9.6}, dE = {:>9.2e}",
            t,
            y[0],
            y[1],
            energy(y) - e0
        );
    }
}
