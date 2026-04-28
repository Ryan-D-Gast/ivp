//! Harmonic oscillator solved with a fixed-step symplectic method.

use ivp::prelude::*;

struct HarmonicOscillator;

impl SecondOrderSystem for HarmonicOscillator {
    fn acceleration(&self, _t: f64, q: &[f64], _p: &[f64], a: &mut [f64]) {
        a[0] = -q[0];
    }
}

fn energy(state: &[f64]) -> f64 {
    0.5 * (state[0] * state[0] + state[1] * state[1])
}

fn main() {
    let q0 = [1.0];
    let v0 = [0.0];
    let t_eval: Vec<f64> = (0..=20).map(|i| i as f64).collect();

    match Ivp::second_order(&HarmonicOscillator, 0.0, 20.0, &q0, &v0)
        .method(SymplecticMethod::VelocityVerlet)
        .step_size(0.05)
        .t_eval(t_eval)
        .solve()
    {
        Ok(sol) => {
            println!("Status: {:?}", sol.status);
            println!("nfev: {}, steps: {}\n", sol.nfev, sol.nstep);

            let e0 = energy(&sol.y[0]);
            for (t, y) in sol.iter() {
                let e = energy(y);
                println!(
                    "t={:>5.2}: q={:>9.6}, v={:>9.6}, dE={:>9.2e}",
                    t,
                    y[0],
                    y[1],
                    e - e0
                );
            }
        }
        Err(e) => eprintln!("Error: {:?}", e),
    }
}
