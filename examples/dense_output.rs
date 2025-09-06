//! Example: Dense output interpolation on a harmonic oscillator

use ivp::prelude::*;
use std::f64::consts::PI;

struct SHO;

impl ODE for SHO {
    fn ode(&self, _x: f64, y: &[f64], dydx: &mut [f64]) {
        // y' = [y1, -y0]
        dydx[0] = y[1];
        dydx[1] = -y[0];
    }
}

fn main() {
    let f = SHO;
    let x0 = 0.0;
    let xend = 2.0 * PI; // one period
    let y0 = [1.0, 0.0];

    let options = IVPOptions::builder()
        .method(Method::RK45)
        .rtol(1e-9)
        .atol(1e-9)
        .dense_output(true)
        .build();

    let sol = solve_ivp(&f, x0, xend, &y0, options).expect("solve_ivp failed");
    println!("Final status: {:?}", sol.status);
    println!("Steps: {} (accepted {} / rejected {})", sol.nstep, sol.naccpt, sol.nrejct);

    // Use dense_output to evaluate on a fine grid
    if let Some(dense) = sol.dense_output.as_ref() {
        let (t0, t1) = dense.t_span().unwrap_or((x0, xend));
        let npts = 41;
        let ts: Vec<f64> = (0..=npts)
            .map(|i| t0 + (t1 - t0) * (i as f64) / (npts as f64))
            .collect();
        let ys = dense.evaluate_many(&ts);

        // Print a few interpolated samples and the analytic reference
        for (i, (t, y_opt)) in ts.iter().zip(ys.iter()).enumerate() {
            if i % 8 == 0 {
                if let Some(y) = y_opt {
                    let y_ref0 = (t).cos();
                    let y_ref1 = -(t).sin();
                    println!(
                        "t = {:>7.4}, y = [{:>.6}, {:>.6}]  ref = [{:>.6}, {:>.6}]",
                        t, y[0], y[1], y_ref0, y_ref1
                    );
                }
            }
        }
    } else {
        println!("dense_output was not enabled");
    }
}
