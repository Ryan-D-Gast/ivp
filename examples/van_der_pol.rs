//! Stiff Van der Pol oscillator solved with BDF method.

use ivp::prelude::*;

struct VanDerPol {
    eps: f64,
}

impl IVP for VanDerPol {
    fn ode(&self, _t: f64, y: &[f64], dydt: &mut [f64]) {
        dydt[0] = y[1];
        dydt[1] = ((1.0 - y[0] * y[0]) * y[1] - y[0]) / self.eps;
    }
}

fn main() {
    let eps = 1e-3;
    let vdp = VanDerPol { eps };
    let y0 = [2.0, 0.0];
    let t_eval: Vec<f64> = (0..=20).map(|i| i as f64 * 0.1).collect();

    let options = Options::builder()
        .method(Method::BDF)
        .rtol(1e-6)
        .atol(1e-8)
        .t_eval(t_eval)
        .build();

    match solve_ivp(&vdp, 0.0, 2.0, &y0, options) {
        Ok(sol) => {
            println!("Status: {:?}", sol.status);
            println!("nfev: {}, njev: {}, nlu: {}", sol.nfev, sol.njev, sol.nlu);

            for (t, y) in sol.iter() {
                println!("t={:.2}: y0={:>9.5}, y1={:>9.5}", t, y[0], y[1]);
            }
        }
        Err(e) => eprintln!("Error: {:?}", e),
    }
}
