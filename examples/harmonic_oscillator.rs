//! Example demonstrating the use of RK23 for solving a harmonic oscillator.

use ivp::prelude::*;
use std::f64::consts::PI;

struct HarmonicOscillator;

impl ODE for HarmonicOscillator {
    fn ode(&self, _x: f64, y: &[f64], dydx: &mut [f64]) {
        dydx[0] = y[1];
        dydx[1] = -y[0];
    }
}

// No-op SolOut to satisfy generics when not using a custom callback
struct NoOpSolOut;
impl SolOut for NoOpSolOut {
    fn solout<I: Interpolate>(
        &mut self,
        _xold: f64,
        _x: f64,
        _y: &[f64],
        _interpolator: &I,
    ) -> ControlFlag {
        ControlFlag::Continue
    }
}

fn main() {
    let harmonic_oscillator = HarmonicOscillator;
    let x0 = 0.0;
    let y0 = [1.0, 0.0];
    let xend = 2.0 * PI;
    let t_eval: Vec<f64> = (0..=20).map(|i| i as f64 * (PI / 10.0)).collect();
    let options = IVPOptions::<NoOpSolOut>::builder()
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

            for (ti, yi) in sol.t.iter().zip(sol.y.iter()) {
                println!("x = {:>8.5}, y = {:?}", ti, yi);
            }
        }
        Err(err) => eprintln!("Integration failed: {:?}", err),
    }
}
