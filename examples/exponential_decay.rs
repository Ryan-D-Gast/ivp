//! Exponential decay: dy/dt = -k*y with analytical comparison.

use ivp::prelude::*;

struct ExponentialDecay {
    k: f64,
}

impl FirstOrderSystem for ExponentialDecay {
    fn derivative(&self, _t: f64, y: &[f64], _p: &[f64], dydt: &mut [f64]) {
        dydt[0] = -self.k * y[0];
    }
}

fn main() {
    let k = 0.5;
    let decay = ExponentialDecay { k };
    let y0 = [10.0];
    let t_eval: Vec<f64> = (0..=10).map(|i| i as f64).collect();

    match Ivp::first_order(&decay, 0.0, 10.0, &y0)
        .method(Method::DOPRI5)
        .rtol(1e-8)
        .atol(1e-10)
        .t_eval(t_eval)
        .solve()
    {
        Ok(sol) => {
            println!("Status: {:?}", sol.status);
            println!("nfev: {}, steps: {}\n", sol.nfev, sol.nstep);

            let mut max_error = 0.0_f64;
            for (t, y_num) in sol.iter() {
                let y_exact = y0[0] * (-k * t).exp();
                let error = (y_num[0] - y_exact).abs();
                max_error = max_error.max(error);
                println!(
                    "t={:.1}: y={:.8}, exact={:.8}, err={:.2e}",
                    t, y_num[0], y_exact, error
                );
            }
            println!("\nMax error: {:.2e}", max_error);
        }
        Err(e) => eprintln!("Error: {:?}", e),
    }
}
