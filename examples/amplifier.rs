//! # Example: Amplifier DAE (index-1)
//!
//! Based on Hairer & Wanner, Solving ODEs II. Demonstrates the mass-matrix
//! formulation `M y' = f(t,y)` with a singular M.

use ivp::prelude::*;
use std::f64::consts::PI;

struct Amplifier {
    // Circuit parameters
    ue: f64, // input amplitude
    ub: f64, // supply voltage
    uf: f64, // thermal voltage
    alpha: f64,
    beta: f64,
    // Resistors
    r0: f64,
    r1: f64,
    r2: f64,
    r3: f64,
    r4: f64,
    r5: f64,
    r6: f64,
    r7: f64,
    r8: f64,
    r9: f64,
    // Capacitors
    c1: f64,
    c2: f64,
    c3: f64,
    c4: f64,
    c5: f64,
}

impl Amplifier {
    fn new() -> Self {
        Self {
            ue: 0.1,
            ub: 6.0,
            uf: 0.026,
            alpha: 0.99,
            beta: 1.0e-6,
            r0: 1_000.0,
            r1: 9_000.0,
            r2: 9_000.0,
            r3: 9_000.0,
            r4: 9_000.0,
            r5: 9_000.0,
            r6: 9_000.0,
            r7: 9_000.0,
            r8: 9_000.0,
            r9: 9_000.0,
            c1: 1.0e-6,
            c2: 2.0e-6,
            c3: 3.0e-6,
            c4: 4.0e-6,
            c5: 5.0e-6,
        }
    }
}

impl IVP for Amplifier {
    fn ode(&self, t: f64, y: &[f64], f: &mut [f64]) {
        let w = 2.0 * PI * 100.0;
        let uet = self.ue * (w * t).sin();

        // Nonlinear diodes
        let fac1 = self.beta * (((y[3] - y[2]) / self.uf).exp() - 1.0);
        let fac2 = self.beta * (((y[6] - y[5]) / self.uf).exp() - 1.0);

        f[0] = y[0] / self.r9;
        f[1] = (y[1] - self.ub) / self.r8 + self.alpha * fac1;
        f[2] = y[2] / self.r7 - fac1;
        f[3] = y[3] / self.r5 + (y[3] - self.ub) / self.r6 + (1.0 - self.alpha) * fac1;
        f[4] = (y[4] - self.ub) / self.r4 + self.alpha * fac2;
        f[5] = y[5] / self.r3 - fac2;
        f[6] = y[6] / self.r1 + (y[6] - self.ub) / self.r2 + (1.0 - self.alpha) * fac2;
        f[7] = (y[7] - uet) / self.r0;
    }

    fn jac(&self, _t: f64, y: &[f64], j: &mut Matrix) {
        let g14 = self.beta * ((y[3] - y[2]) / self.uf).exp() / self.uf;
        let g27 = self.beta * ((y[6] - y[5]) / self.uf).exp() / self.uf;

        j[(0, 0)] = 1.0 / self.r9;

        j[(1, 1)] = 1.0 / self.r8;
        j[(1, 3)] = self.alpha * g14; // d f1 / d y3
        j[(1, 2)] = -self.alpha * g14; // d f1 / d y2

        j[(2, 2)] = 1.0 / self.r7 + g14;
        j[(2, 3)] = -g14;

        j[(3, 3)] = 1.0 / self.r5 + 1.0 / self.r6 + (1.0 - self.alpha) * g14;
        j[(3, 2)] = -(1.0 - self.alpha) * g14;

        j[(4, 4)] = 1.0 / self.r4;
        j[(4, 6)] = self.alpha * g27; // d f4 / d y6
        j[(4, 5)] = -self.alpha * g27; // d f4 / d y5

        j[(5, 5)] = 1.0 / self.r3 + g27;
        j[(5, 6)] = -g27;

        j[(6, 6)] = 1.0 / self.r1 + 1.0 / self.r2 + (1.0 - self.alpha) * g27;
        j[(6, 5)] = -(1.0 - self.alpha) * g27;

        j[(7, 7)] = 1.0 / self.r0;
    }

    fn mass(&self, m: &mut Matrix) {
        // Diagonal
        m[(0, 0)] = -self.c5;
        m[(1, 1)] = -self.c5;
        m[(2, 2)] = -self.c4;
        m[(3, 3)] = -self.c3;
        m[(4, 4)] = -self.c3;
        m[(5, 5)] = -self.c2;
        m[(6, 6)] = -self.c1;
        m[(7, 7)] = -self.c1;
        // Super-diagonal
        m[(0, 1)] = self.c5;
        m[(3, 4)] = self.c3;
        m[(6, 7)] = self.c1;
        // Sub-diagonal
        m[(1, 0)] = self.c5;
        m[(4, 3)] = self.c3;
        m[(7, 6)] = self.c1;
    }
}

fn main() {
    let model = Amplifier::new();

    // Initial conditions (approximate steady-state)
    let y0 = [
        0.0,
        model.ub, // node 1 near supply
        model.ub / (model.r6 / model.r5 + 1.0),
        model.ub / (model.r6 / model.r5 + 1.0),
        model.ub,
        model.ub / (model.r2 / model.r1 + 1.0),
        model.ub / (model.r2 / model.r1 + 1.0),
        0.0,
    ];

    let x0 = 0.0;
    let xend = 0.05; // 50 ms

    // Output every 2.5 ms
    let t_eval: Vec<f64> = (0..=20).map(|i| i as f64 * 0.0025).collect();

    let options = Options::builder()
        .method(Method::RADAU)
        .mass_storage(MatrixStorage::Banded { ml: 1, mu: 1 })
        .jac_storage(MatrixStorage::Banded { ml: 3, mu: 3 })
        .rtol(1.0e-5)
        .atol(1.0e-11)
        .t_eval(t_eval)
        .build();

    match solve_ivp(&model, x0, xend, &y0, options) {
        Ok(sol) => {
            println!("Finished status: {:?}", sol.status);
            println!(
                "nfev={}, njev={}, nlu={}, nstep={}, naccpt={}, nrejct={}",
                sol.nfev, sol.njev, sol.nlu, sol.nstep, sol.naccpt, sol.nrejct
            );
            println!("{:<10}  {:>20}  {:>20}", "t", "y[0]", "y[1]");
            for (t, y) in sol.iter() {
                println!("{t:10.5}  {y0:>20.10e}  {y1:>20.10e}", y0 = y[0], y1 = y[1]);
            }
        }
        Err(e) => eprintln!("solve_ivp failed: {:?}", e),
    }
}
