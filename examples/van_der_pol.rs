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

use ivp::dp::*;
use ivp::*;

struct VanDerPol {
    eps: f64,
}

impl ODE for VanDerPol {
    fn ode(&self, _x: f64, y: &[f64], dydx: &mut [f64]) {
        dydx[0] = y[1];
        dydx[1] = ((1.0 - y[0] * y[0]) * y[1] - y[0]) / self.eps;
    }
}

struct Printer {
    xout: f64,
    dx: f64,
    first_call: bool,
    xend: f64,
}

impl Printer {
    fn new(dx: f64, xend: f64) -> Self {
        Self {
            xout: 0.0,
            dx,
            first_call: true,
            xend,
        }
    }
}

impl SolOut for Printer {
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
            let mut yi = [0.0; 2];
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
    let mut vdp = VanDerPol { eps: 1e-3 };
    let x0 = 0.0;
    let xend = 2.0;
    let y0 = [2.0, 0.0];

    let settings = Settings::default();
    let mut printer = Printer::new(0.1, xend);

    let rtol = 1e-9;
    let atol = 1e-9;

    let res = dop853(&mut vdp, x0, &y0, xend, rtol, atol, &mut printer, settings);

    match res {
        Ok(r) => {
            println!("Finished status: {:?}", r.status);
            println!("Final State: x = {:.5}, y = {:?}", r.x, r.y);
            println!("Number of function evaluations: {}", r.nfev);
            println!("Number of steps taken: {}", r.nstep);
            println!("Number of accepted steps: {}", r.naccpt);
            println!("Number of rejected steps: {}", r.nrejct);
        }
        Err(e) => eprintln!("dop853 failed: {:?}", e),
    }
}
