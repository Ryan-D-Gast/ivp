//! Example demonstrating the use of RK23 for solving a harmonic oscillator.

use ivp::{Float, ODE, rk::RKSettings, rk::rk23};
use ivp::{ControlFlag, SolOut};
use std::f64::consts::PI;

struct HarmonicOscillator;

impl ODE for HarmonicOscillator {
    fn ode(&self, _x: Float, y: &[Float], dydx: &mut [Float]) {
        dydx[0] = y[1];
        dydx[1] = -y[0];
    }
}

struct EvenOutput {
    xout: Float,
    dx: Float,
    first_call: bool,
    xend: Float,
}

impl EvenOutput {
    fn new(dx: Float, xend: Float) -> Self {
        Self {
            xout: 0.0,
            dx,
            first_call: true,
            xend,
        }
    }
}

impl SolOut for EvenOutput {
    fn solout<I: ivp::Interpolate>(
        &mut self,
        xold: Float,
        x: Float,
        y: &[Float],
        interpolator: &I,
    ) -> ControlFlag {
        if self.first_call {
            println!("x = {:>8.5}, y = {:?}", xold, y);
            self.first_call = false;
            self.xout = xold + self.dx;
        }

        let tol = 1e-12;
        while self.xout <= x + tol {
            let mut yi = vec![0.0; y.len()];
            interpolator.interpolate(self.xout, &mut yi);
            println!("x = {:>8.5}, y = {:?}", self.xout, yi);
            self.xout += self.dx;
        }

        if (x - self.xend).abs() <= tol {
            let last = self.xout - self.dx;
            if (last - x).abs() > tol {
                println!("x = {:>8.5}, y = {:?}", x, y);
            }
        }

        ControlFlag::Continue
    }
}

fn main() {
    let mut ode = HarmonicOscillator;
    let x0 = 0.0;
    let y0 = vec![1.0, 0.0]; // Initial conditions: y[0] = position, y[1] = velocity
    let xend = 2.0 * PI; // One full period of the oscillator
    let rtol = 1e-3;
    let atol = 1e-3;
    let mut solout = EvenOutput::new(PI / 10.0, xend);
    let settings = RKSettings::new();

    match rk23(&mut ode, x0, &y0, xend, rtol, atol, &mut solout, settings) {
        Ok(result) => {
            println!("Integration successful: x = {:.5}, y = {:?}", result.x, result.y);
        }
        Err(err) => {
            eprintln!("Integration failed: {:?}", err);
        }
    }
}
