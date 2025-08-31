//! Example of using the RK4 integrator for a non-stiff ODE

use ivp::rk::{rk4, RKSettings};
use ivp::{Float, ODE, SolOut, ControlFlag, Interpolate};

struct SimpleODE;

impl<const N: usize> ODE<N> for SimpleODE {
    fn ode(&mut self, _x: Float, y: &[Float; N], dydx: &mut [Float; N]) {
        // Example: dy/dx = -y (exponential decay)
        for i in 0..N {
            dydx[i] = -y[i];
        }
    }
}

struct Printer;

impl<const N: usize> SolOut<N> for Printer {
    fn solout<I: Interpolate<N>>(
        &mut self,
        _xold: Float,
        x: Float,
        y: &[Float; N],
        _interpolator: &I,
    ) -> ControlFlag {
        println!("x = {:.4}, y = {:?}", x, y);
        ControlFlag::Continue
    }
}

fn main() {
    let mut f = SimpleODE;
    let mut solout = Printer;

    let x0 = 0.0;
    let xend = 5.0;
    let y0 = [1.0];
    let h = 0.1;

    let settings = RKSettings::new();

    match rk4(&mut f, x0, y0, xend, h, &mut solout, settings) {
        Ok(result) => {
            println!("Integration successful: x = {:.4}, y = {:?}", result.x, result.y);
        }
        Err(e) => {
            println!("Integration failed: {}", e);
        }
    }
}
