//! # Example: Exponential Decay
//!
//! Solve the exponential decay equation as a first-order system.
//!
//! Equations:
//! dy/dx = -y
//!
//! Initial condition: y(0) = 1.0
//!

use ivp::rk::rk4;
use ivp::{ControlFlag, Float, Interpolate, ODE, Settings, SolOut};

struct SimpleODE;

impl ODE for SimpleODE {
    fn ode(&self, _x: Float, y: &[Float], dydx: &mut [Float]) {
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
        _xold: Float,
        x: Float,
        y: &[Float],
        _interpolator: &I,
    ) -> ControlFlag {
        println!("x = {:.4}, y = {:?}", x, y);
        ControlFlag::Continue
    }
}

fn main() {
    let f = SimpleODE;
    let mut solout = Printer;

    let x0 = 0.0;
    let xend = 5.0;
    let y0 = [1.0];
    let h = 0.1;

    let settings = Settings::builder()
        .build();

    match rk4(&f, x0, &y0, xend, h, &mut solout, settings) {
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
