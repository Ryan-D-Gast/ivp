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

struct Printer;

impl SolOut for Printer {
    fn solout<I: Interpolate>(
        &mut self,
        _xold: f64,
        x: f64,
        y: &[f64],
        _interpolator: &I,
    ) -> ControlFlag {
        println!("x = {:.4}, y = {:?}", x, y);
        ControlFlag::Continue
    }
}

fn main() {
    let f = SimpleODE;

    let x0 = 0.0;
    let xend = 5.0;
    let y0 = [1.0];
    let h = 0.1;

    let settings = Settings::builder().build();

    match rk4(&f, x0, xend, &y0, h, Some(&mut Printer), settings) {
        Ok(result) => {
            println!("Final status: {:?}", result.status);
            println!("Final state: x = {:.5}, y = {:?}", result.x, result.y);
            println!("Number of function evaluations: {}", result.nfev);
            println!("Number of steps taken: {}", result.nstep);
            println!("Number of accepted steps: {}", result.naccpt);
            println!("Number of rejected steps: {}", result.nrejct);
        }
        Err(e) => {
            println!("Integration failed: {}", e);
        }
    }
}
