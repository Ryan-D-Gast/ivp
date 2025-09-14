use ivp::prelude::*;

mod common;
use common::{SHO, default_opts};

fn methods() -> Vec<Method> {
    vec![Method::RK4, Method::RK23, Method::DOPRI5, Method::DOP853, Method::Radau5]
}

#[test]
fn harmonic_accuracy_end_state() {
    let x0 = 0.0;
    let xend = 2.0 * std::f64::consts::PI; // one period
    let y0 = [1.0, 0.0];

    for method in methods() {
        let method_dbg = format!("{:?}", method);
        let opts = if let Method::RK4 = method {
            // fixed step RK4: choose step to land on period
            let h = (xend - x0) / 2000.0;
            Options::builder()
                .method(method.clone())
                .first_step(h)
                .build()
        } else {
            default_opts(method.clone())
        };
        let sol = solve_ivp(&SHO, x0, xend, &y0, opts).expect("solve_ivp failed");
        let y_end = sol.y.last().unwrap().clone();
        assert!(
            (y_end[0] - 1.0).abs() < 1e-5,
            "cos end mismatch for {}",
            method_dbg
        );
        assert!(
            y_end[1].abs() < 1e-5,
            "sin' end mismatch for {}",
            method_dbg
        );
    }
}

#[test]
fn t_eval_sampling_exact_times() {
    let x0 = 0.0;
    let xend = 1.0;
    let y0 = [1.0, 0.0];
    let t_eval: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();

    for method in methods() {
        let method_dbg = format!("{:?}", method);
        let opts = Options::builder()
            .method(method)
            .rtol(1e-9)
            .atol(1e-9)
            .t_eval(t_eval.clone())
            .build();
        let sol = solve_ivp(&SHO, x0, xend, &y0, opts).expect("solve_ivp failed");
        let tol = 1e-9;
        for &te in &t_eval {
            assert!(
                sol.t.iter().any(|&t| (t - te).abs() <= tol),
                "t_eval not respected for {} (missing {})",
                method_dbg,
                te
            );
        }
        assert_eq!(sol.y.len(), sol.t.len());
    }
}

#[test]
fn iterate_samples() {
    let x0 = 0.0;
    let xend = 1.0;
    let y0 = [1.0, 0.0];
    let sol = solve_ivp(&SHO, x0, xend, &y0, default_opts(Method::DOPRI5)).unwrap();
    for (t, y) in sol.iter() {
        assert!(t >= x0 && t <= xend);
        assert_eq!(y.len(), y0.len());
    }
}
