//! # Example: Exponential Decay
//!
//! Solve the exponential decay equation as a first-order system.
//!
//! Equations:
//! dy/dx = -y
//!
//! Initial condition: y(0) = 1.0
//! 

use ivp::rk::{rk4, RKSettings};
use ivp::{Float, ODE, SolOut, ControlFlag, Interpolate};

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

    let settings = RKSettings::new();

    match rk4(&f, x0, &y0, xend, h, &mut solout, settings) {
        Ok(result) => {
            println!("Integration successful: x = {:.4}, y = {:?}", result.x, result.y);
        }
        Err(e) => {
            println!("Integration failed: {}", e);
        }
    }
}
