//! # Example: Exponential Decay
//!
//! Solve the exponential decay equation as a first-order system.
//!
//! Equations:
//! dy/dx = -y
//!
//! Initial condition: y(0) = 1.0
//!

use ivp::prelude::*;

struct SimpleODE;

impl ODE for SimpleODE {
    fn ode(&self, _x: f64, y: &[f64], dydx: &mut [f64]) {
        // Example: dy/dx = -y (exponential decay)
        for i in 0..y.len() {
            dydx[i] = -y[i];
        }
    }
}

fn main() {
    let f = SimpleODE;
    let x0 = 0.0;
    let xend = 5.0;
    let y0 = [1.0];
    let t_eval: Vec<f64> = (0..=50).map(|i| i as f64 * 0.1).collect();

    let options = Options::builder()
        // Default method is DOPRI5 (Also known as RK45 in SciPy)
        .rtol(1e-6)
        .atol(1e-6)
        .t_eval(t_eval)
        .build();

    match solve_ivp(&f, x0, xend, &y0, options) {
        Ok(sol) => {
            println!("Final status: {:?}", sol.status);
            if let (Some(&t_last), Some(y_last)) = (sol.t.last(), sol.y.last()) {
                println!("Final state: x = {:.5}, y = {:?}", t_last, y_last);
            }
            println!("Number of function evaluations: {}", sol.nfev);
            println!("Number of steps taken: {}", sol.nstep);
            println!("Number of accepted steps: {}", sol.naccpt);
            println!("Number of rejected steps: {}", sol.nrejct);

            // Print sampled values
            for (ti, yi) in sol.t.iter().zip(sol.y.iter()) {
                println!("x = {:.4}, y = {:?}", ti, yi);
            }
        }
        Err(e) => eprintln!("Integration failed: {:?}", e),
    }
}
