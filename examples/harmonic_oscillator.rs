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

struct EvenOutput {
    xout: f64,
    dx: f64,
    first_call: bool,
    xend: f64,
}

impl EvenOutput {
    fn new(dx: f64, xend: f64) -> Self {
        Self {
            xout: 0.0,
            dx,
            first_call: true,
            xend,
        }
    }
}

impl SolOut for EvenOutput {
    fn solout<I: Interpolate>(
        &mut self,
        xold: f64,
        x: f64,
        y: &[f64],
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
    let harmonic_oscillator = HarmonicOscillator;
    let x0 = 0.0;
    let y0 = [1.0, 0.0];
    let xend = 2.0 * PI;
    let settings = Settings::builder().rtol(1e-3).atol(1e-3).build();

    match rk23(
        &harmonic_oscillator,
        x0,
        xend,
        &y0,
        Some(&mut EvenOutput::new(PI / 10.0, xend)),
        settings,
    ) {
        Ok(result) => {
            println!("Final status: {:?}", result.status);
            println!("Final state: x = {:.5}, y = {:?}", result.x, result.y);
            println!("Number of function evaluations: {}", result.nfev);
            println!("Number of steps taken: {}", result.nstep);
            println!("Number of accepted steps: {}", result.naccpt);
            println!("Number of rejected steps: {}", result.nrejct);
        }
        Err(err) => {
            eprintln!("Integration failed: {:?}", err);
        }
    }
}
