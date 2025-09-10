//! # Example: Van der Pol oscillator
//!
//! Solve the stiff Van der Pol oscillator as a first-order system.
//!
//! Equations:
//! dy0/dt = y1
//! dy1/dt = ((1 - y0^2) * y1 - y0) / mu
//!
//! Initial conditions: y0(0) = 2.0, y1(0) = 0.0
//!

use ivp::prelude::*;

struct VanDerPol {
    eps: f64,
}

impl ODE for VanDerPol {
    fn ode(&self, _x: f64, y: &[f64], dydx: &mut [f64]) {
        dydx[0] = y[1];
        dydx[1] = ((1.0 - y[0] * y[0]) * y[1] - y[0]) / self.eps;
    }
}

fn main() {
    let van_der_pol = VanDerPol { eps: 1e-3 };
    let x0 = 0.0;
    let xend = 2.0;
    let y0 = [2.0, 0.0];
    let t_eval = (0..=20).map(|i| i as f64 * 0.1).collect();
    let options = Options::builder()
        .method(Method::DOP853)
        .rtol(1e-9)
        .atol(1e-9)
        .t_eval(t_eval)
        .build();

    match solve_ivp(&van_der_pol, x0, xend, &y0, options) {
        Ok(sol) => {
            println!("Finished status: {:?}", sol.status);
            if let (Some(&t_last), Some(y_last)) = (sol.t.last(), sol.y.last()) {
                println!("Final State: x = {:.5}, y = {:?}", t_last, y_last);
            }
            println!("Number of function evaluations: {}", sol.nfev);
            println!("Number of steps taken: {}", sol.nstep);
            println!("Number of accepted steps: {}", sol.naccpt);
            println!("Number of rejected steps: {}", sol.nrejct);

            for (ti, yi) in sol.t.iter().zip(sol.y.iter()) {
                println!("x = {:>8.5}, y = {:?}", ti, yi);
            }
        }
        Err(e) => eprintln!("solve_ivp failed: {:?}", e),
    }
}
