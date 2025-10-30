//! # Example: Harmonic Oscillator
//!
//! Solve the harmonic oscillator as a first-order system.
//!
//! Equations:
//! dy0/dt = y1
//! dy1/dt = -y0
//!
//! Initial conditions: y0(0) = 1, y1(0) = 0
//!

use ivp::prelude::*;
use std::f64::consts::PI;

struct HarmonicOscillator;

impl IVP for HarmonicOscillator {
    fn ode(&self, _x: f64, y: &[f64], dydx: &mut [f64]) {
        dydx[0] = y[1];
        dydx[1] = -y[0];
    }
}

fn main() {
    let harmonic_oscillator = HarmonicOscillator;
    let x0 = 0.0;
    let y0 = [1.0, 0.0];
    let xend = 2.0 * PI;
    let t_eval = (0..=20).map(|i| i as f64 * (PI / 10.0)).collect();
    let options = Options::builder()
        .method(Method::RK23)
        .rtol(1e-3)
        .atol(1e-3)
        .t_eval(t_eval)
        .build();

    match solve_ivp(&harmonic_oscillator, x0, xend, &y0, options) {
        Ok(sol) => {
            println!("Final status: {:?}", sol.status);
            if let (Some(&t_last), Some(y_last)) = (sol.t.last(), sol.y.last()) {
                println!("Final state: x = {:.5}, y = {:?}", t_last, y_last);
            }
            println!("Number of function evaluations: {}", sol.nfev);
            println!("Number of steps taken: {}", sol.nstep);
            println!("Number of accepted steps: {}", sol.naccpt);
            println!("Number of rejected steps: {}", sol.nrejct);

            for (ti, yi) in sol.iter() {
                println!("x = {:>8.5}, y = {:?}", ti, yi);
            }
        }
        Err(err) => eprintln!("Integration failed: {:?}", err),
    }
}
