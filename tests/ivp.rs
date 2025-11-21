use ivp::prelude::*;

mod common;
use common::{SHO, default_opts, default_opts_dense};

fn all_methods() -> Vec<Method> {
    vec![Method::RK23, Method::DOPRI5, Method::DOP853, Method::RADAU, Method::BDF]
}

#[derive(Clone, Copy)]
struct ZeroRhs;
impl IVP for ZeroRhs {
    fn ode(&self, _t: f64, _y: &[f64], dydx: &mut [f64]) {
        for v in dydx.iter_mut() {
            *v = 0.0;
        }
    }
}

#[test]
fn integration_zero_rhs_all_methods() {
    let f = ZeroRhs;
    let x0 = 0.0;
    let xend = 10.0;
    let y0 = vec![1.0, 1.0, 1.0];
    let t_eval: Vec<f64> = (0..=20)
        .map(|i| x0 + (xend - x0) * (i as f64) / 20.0)
        .collect();
    // BDF is not suitable for RHS=0 and is thus excluded
    let all_but_bdf = all_methods().iter().filter(|m| **m != Method::BDF).cloned().collect::<Vec<_>>();
    for method in all_but_bdf {
        let opts = Options::builder()
            .method(method.clone())
            .rtol(1e-9)
            .atol(1e-12)
            .t_eval(t_eval.clone())
            .build();
        let sol = solve_ivp(&f, x0, xend, &y0, opts).expect("solve_ivp failed");
        assert_eq!(sol.t, t_eval);
        for yi in sol.y {
            for &v in &yi {
                assert!((v - 1.0).abs() <= 1e-12);
            }
        }
    }
}

#[test]
fn max_step_and_first_step_controls() {
    let x0 = 0.0;
    let xend = 3.0;
    let y0 = [1.0, 0.0];

    // max_step respected
    let max_step = 0.05;
    for method in all_methods() {
        let opts = Options::builder()
            .method(method.clone())
            .rtol(1e-6)
            .atol(1e-9)
            .max_step(max_step)
            .build();
        let sol = solve_ivp(&SHO, x0, xend, &y0, opts).expect("solve_ivp failed");
        for w in sol.t.windows(2) {
            let dt = (w[1] - w[0]).abs();
            assert!(
                dt <= max_step + 1e-12,
                "dt={} exceeds max_step {} for {:?}",
                dt,
                max_step,
                method
            );
        }
    }

    // first_step honored (check the first actual step size)
    let first_step = 0.1;
    // TODO: BDF modifies the first step internally, 
    // needs to be resolved with the solout callback instead.
    // For now we exclude it from this test.
    for method in all_methods() {
        let opts = Options::builder()
            .method(method.clone())
            .rtol(1e-3)
            .atol(1e-6)
            .first_step(first_step)
            .build();
        let sol = solve_ivp(&SHO, x0, xend, &y0, opts).expect("solve_ivp failed");
        assert!(
            sol.t.len() >= 2,
            "expected at least two time points for {:?}",
            method
        );
        let dt0 = (sol.t[1] - sol.t[0]).abs();
        assert!(
            (dt0 - first_step).abs() <= 1e-6,
            "first step mismatch: got {}, want {} for {:?}",
            dt0,
            first_step,
            method
        );
    }
}

#[test]
fn dense_output_matches_discrete_samples() {
    let x0 = 0.0;
    let xend = 2.0;
    let y0 = [1.0, 0.0];
    for method in [Method::RK23, Method::DOPRI5, Method::DOP853, Method::RADAU] {
        let opts = Options::builder()
            .method(method.clone())
            .rtol(1e-8)
            .atol(1e-10)
            .dense_output(true)
            .build();
        let sol = solve_ivp(&SHO, x0, xend, &y0, opts).expect("solve_ivp failed");
        // Ensure span exists
        assert!(sol.sol_span().is_some());
        // Evaluate dense output at stored times and compare
        let ys_dense = sol.sol_many(&sol.t).expect("dense evaluation failed");
        assert_eq!(ys_dense.len(), sol.y.len());
        for (yd, ys) in ys_dense.iter().zip(sol.y.iter()) {
            assert_eq!(yd.len(), ys.len());
            for (a, b) in yd.iter().zip(ys.iter()) {
                assert!(
                    (a - b).abs() <= 1e-8,
                    "dense vs stored mismatch: {} vs {}",
                    a,
                    b
                );
            }
        }
    }
}

#[test]
fn dense_output_out_of_range_errors() {
    let x0 = 0.0;
    let xend = 1.0;
    let y0 = [1.0, 0.0];
    let sol = solve_ivp(&SHO, x0, xend, &y0, default_opts_dense(Method::DOPRI5)).unwrap();
    let (t0, t1) = sol.sol_span().unwrap();
    let before = t0 - 0.1;
    let after = t1 + 0.1;
    assert!(sol.sol(before).is_err());
    assert!(sol.sol(after).is_err());
}

struct ShoZeroEventAll;
impl IVP for ShoZeroEventAll {
    fn ode(&self, _t: f64, y: &[f64], dydx: &mut [f64]) {
        dydx[0] = y[1];
        dydx[1] = -y[0];
    }
    fn events(&self, _t: f64, y: &[f64], out: &mut [f64]) {
        out[0] = y[0];
    }
    fn n_events(&self) -> usize {
        1
    }
    fn event_config(&self, _index: usize) -> EventConfig {
        let mut cfg = EventConfig::default();
        // Detect all zero-crossings, stop after 2
        cfg.all();
        cfg.terminal_count(2);
        cfg
    }
}

struct ShoZeroEventPositive;
impl IVP for ShoZeroEventPositive {
    fn ode(&self, _t: f64, y: &[f64], dydx: &mut [f64]) {
        dydx[0] = y[1];
        dydx[1] = -y[0];
    }
    fn events(&self, _t: f64, y: &[f64], out: &mut [f64]) {
        out[0] = y[0];
    }
    fn n_events(&self) -> usize {
        1
    }
    fn event_config(&self, _index: usize) -> EventConfig {
        let mut cfg = EventConfig::default();
        cfg.positive();
        cfg.terminal();
        cfg
    }
}

struct ShoZeroEventNegative;
impl IVP for ShoZeroEventNegative {
    fn ode(&self, _t: f64, y: &[f64], dydx: &mut [f64]) {
        dydx[0] = y[1];
        dydx[1] = -y[0];
    }
    fn events(&self, _t: f64, y: &[f64], out: &mut [f64]) {
        out[0] = y[0];
    }
    fn n_events(&self) -> usize {
        1
    }
    fn event_config(&self, _index: usize) -> EventConfig {
        let mut cfg = EventConfig::default();
        cfg.negative();
        cfg.terminal();
        cfg
    }
}

fn find_near_zero_times(t: &[f64], y: &[Vec<f64>], tol: f64) -> Vec<f64> {
    let mut hits = Vec::new();
    for (ti, yi) in t.iter().copied().zip(y.iter()) {
        if yi[0].abs() <= tol {
            hits.push(ti);
        }
    }
    hits
}

#[test]
fn event_detection_all_and_directional() {
    let y0 = [1.0, 0.0];
    let x0 = 0.0;
    let xend = 6.0; // covers several zero crossings

    // All crossings, stop after 2 -> expect near pi/2 and 3pi/2, terminates at latter
    let f_all = ShoZeroEventAll;
    let sol_all = solve_ivp(&f_all, x0, xend, &y0, default_opts(Method::DOPRI5)).unwrap();
    let zeros = find_near_zero_times(&sol_all.t_events[0], &sol_all.y_events[0], 1e-8);
    // Expect at least two event samples recorded
    assert!(
        zeros.len() >= 2,
        "expected at least two zero-crossings recorded"
    );
    let pi_over_2 = std::f64::consts::FRAC_PI_2;
    let three_pi_over_2 = 3.0 * std::f64::consts::FRAC_PI_2;
    let z1 = zeros[0];
    let z2 = *zeros.last().unwrap();
    assert!(
        (z1 - pi_over_2).abs() < 5e-3,
        "first zero ~pi/2, got {}",
        z1
    );
    assert!(
        (z2 - three_pi_over_2).abs() < 5e-3,
        "second zero ~3pi/2, got {}",
        z2
    );

    // Positive-going only -> expect ~3pi/2
    let f_pos = ShoZeroEventPositive;
    let sol_pos = solve_ivp(&f_pos, x0, xend, &y0, default_opts(Method::DOPRI5)).unwrap();
    let zeros_pos = find_near_zero_times(&sol_pos.t_events[0], &sol_pos.y_events[0], 1e-8);
    assert!(!zeros_pos.is_empty());
    let z = zeros_pos[0];
    assert!(
        (z - three_pi_over_2).abs() < 5e-3,
        "positive crossing near 3pi/2, got {}",
        z
    );

    // Negative-going only -> expect ~pi/2
    let f_neg = ShoZeroEventNegative;
    let sol_neg = solve_ivp(&f_neg, x0, xend, &y0, default_opts(Method::DOPRI5)).unwrap();
    let zeros_neg = find_near_zero_times(&sol_neg.t_events[0], &sol_neg.y_events[0], 1e-8);
    assert!(!zeros_neg.is_empty());
    let z = zeros_neg[0];
    assert!(
        (z - pi_over_2).abs() < 5e-3,
        "negative crossing near pi/2, got {}",
        z
    );
}

#[test]
fn zero_interval_returns_initial_state() {
    let x0 = 1.23;
    let xend = 1.23;
    let y0 = [2.0, 3.0];
    for method in all_methods() {
        let sol = solve_ivp(&SHO, x0, xend, &y0, default_opts(method)).unwrap();
        assert!(!sol.t.is_empty());
        let y_last = sol.y.last().unwrap();
        assert!((y_last[0] - y0[0]).abs() <= 1e-12);
        assert!((y_last[1] - y0[1]).abs() <= 1e-12);
    }
}

struct Exp2;
impl IVP for Exp2 {
    fn ode(&self, _t: f64, y: &[f64], dydx: &mut [f64]) {
        dydx[0] = y[0];
        dydx[1] = y[1];
    }
}

#[test]
fn vector_rtol_componentwise_control() {
    // y' = y, y(0) = 1 => y(1) = e
    let f = Exp2;
    let x0 = 0.0;
    let xend = 1.0;
    let y0 = [1.0, 1.0];

    // looser vs tighter rtol on second component
    let opts_loose = Options::builder()
        .method(Method::DOPRI5)
        .rtol([1e-2, 1e-2])
        .atol(1e-10)
        .build();
    let opts_tight_second = Options::builder()
        .method(Method::DOPRI5)
        .rtol([1e-2, 1e-10])
        .atol(1e-10)
        .build();

    let sol_loose = solve_ivp(&f, x0, xend, &y0, opts_loose).unwrap();
    let sol_tight = solve_ivp(&f, x0, xend, &y0, opts_tight_second).unwrap();
    let y_end_loose = sol_loose.y.last().unwrap();
    let y_end_tight = sol_tight.y.last().unwrap();
    let exact = std::f64::consts::E;

    let err1_loose = (y_end_loose[0] - exact).abs();
    let err2_loose = (y_end_loose[1] - exact).abs();
    let err1_tight = (y_end_tight[0] - exact).abs();
    let err2_tight = (y_end_tight[1] - exact).abs();

    // tightening the second component should reduce its error
    assert!(err2_tight < err2_loose * 0.5);
    // and should not worsen the first dramatically
    assert!(err1_tight <= 10.0 * err1_loose);
}
