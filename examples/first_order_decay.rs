//! First-order example: exponential decay with an analytical reference.

use ivp::prelude::*;

struct ExponentialDecay {
    rate: f64,
}

impl FirstOrderSystem for ExponentialDecay {
    fn derivative(&self, _t: f64, y: &[f64], _p: &[f64], dydt: &mut [f64]) {
        dydt[0] = -self.rate * y[0];
    }
}

fn main() {
    let system = ExponentialDecay { rate: 0.5 };
    let y0 = [1.0];
    let t_eval: Vec<f64> = (0..=10).map(|i| i as f64).collect();

    let sol = Ivp::first_order(&system, 0.0, 10.0, &y0)
        .method(Method::DOPRI5)
        .rtol(1e-8)
        .atol(1e-10)
        .t_eval(t_eval)
        .solve()
        .unwrap();

    println!("status: {:?}", sol.status);
    for (t, y) in sol.iter() {
        let exact = (-system.rate * t).exp();
        println!(
            "t = {:>4.1}, y = {:>10.7}, exact = {:>10.7}, err = {:.2e}",
            t,
            y[0],
            exact,
            (y[0] - exact).abs()
        );
    }
}
