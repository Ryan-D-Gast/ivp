use ivp::prelude::*;

mod common;
use common::{SHO, default_opts_dense};

#[test]
fn backward_integration_works() {
    let x0 = 2.0 * std::f64::consts::PI;
    let xend = 0.0;
    let y0 = [1.0, 0.0];
    for method in [Method::RK23, Method::DOPRI5, Method::DOP853, Method::Radau5] {
        let sol = solve_ivp(&SHO, x0, xend, &y0, default_opts_dense(method)).unwrap();
        // Check we got a span and can evaluate at mid
        if let Some((t0, t1)) = sol.sol_span() {
            assert!(t0 > t1); // backward span
            let mid = 0.5 * (t0 + t1);
            let y_mid = sol.sol(mid).unwrap();
            let y_ref0 = mid.cos();
            let y_ref1 = -mid.sin();
            assert!((y_mid[0] - y_ref0).abs() < 1e-6);
            assert!((y_mid[1] - y_ref1).abs() < 1e-6);
        } else {
            panic!("no dense span");
        }
    }
}
