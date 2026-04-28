use ivp::error::{ConfigError, Error};
use ivp::prelude::*;

struct HarmonicSecondOrder;

impl SecondOrderSystem for HarmonicSecondOrder {
    fn acceleration(&self, _t: f64, q: &[f64], _p: &[f64], a: &mut [f64]) {
        a[0] = -q[0];
    }
}

struct HarmonicHamiltonian;

impl SeparableHamiltonianSystem for HarmonicHamiltonian {
    fn position_derivative(&self, _t: f64, p: &[f64], _params: &[f64], dqdt: &mut [f64]) {
        dqdt[0] = p[0];
    }

    fn momentum_derivative(&self, _t: f64, q: &[f64], _params: &[f64], dpdt: &mut [f64]) {
        dpdt[0] = -q[0];
    }
}

fn energy(state: &[f64]) -> f64 {
    0.5 * (state[0] * state[0] + state[1] * state[1])
}

fn symplectic_methods() -> [SymplecticMethod; 5] {
    [
        SymplecticMethod::SymplecticEulerKickDrift,
        SymplecticMethod::SymplecticEulerDriftKick,
        SymplecticMethod::VelocityVerlet,
        SymplecticMethod::Ruth3,
        SymplecticMethod::Yoshida4,
    ]
}

#[test]
fn all_methods_reach_end_time() {
    let q0 = [1.0];
    let p0 = [0.0];

    for method in symplectic_methods() {
        let sol = Ivp::hamiltonian(&HarmonicHamiltonian, 0.0, 2.0, &q0, &p0)
            .method(method)
            .step_size(0.05)
            .solve()
            .expect("Ivp::solve failed");

        assert_eq!(sol.status, Status::Success, "{:?}", method);
        assert!((sol.t.last().copied().unwrap() - 2.0).abs() <= 1e-12);
        assert_eq!(sol.y.last().unwrap().len(), 2);
    }
}

#[test]
fn velocity_verlet_second_order_keeps_energy_bounded() {
    let q0 = [1.0];
    let v0 = [0.0];
    let sol = Ivp::second_order(&HarmonicSecondOrder, 0.0, 200.0, &q0, &v0)
        .method(SymplecticMethod::VelocityVerlet)
        .step_size(0.05)
        .solve()
        .expect("Ivp::solve failed");

    assert_eq!(sol.status, Status::Success);
    let e0 = energy(&sol.y[0]);
    let emax = sol
        .y
        .iter()
        .map(|state| energy(state))
        .fold(f64::NEG_INFINITY, f64::max);
    let emin = sol
        .y
        .iter()
        .map(|state| energy(state))
        .fold(f64::INFINITY, f64::min);

    assert!(
        (emax - emin) < 1e-3,
        "energy band too large: {}",
        emax - emin
    );
    assert!((energy(sol.y.last().unwrap()) - e0).abs() < 1e-3);
}

#[test]
fn t_eval_is_sampled_exactly() {
    let q0 = [1.0];
    let p0 = [0.0];
    let t_eval = vec![0.0, 0.13, 0.77, 1.01, 1.73, 2.0];
    let sol = Ivp::hamiltonian(&HarmonicHamiltonian, 0.0, 2.0, &q0, &p0)
        .method(SymplecticMethod::Yoshida4)
        .step_size(0.2)
        .t_eval(t_eval.clone())
        .solve()
        .expect("Ivp::solve failed");

    assert_eq!(sol.status, Status::Success);
    assert_eq!(sol.t.len(), t_eval.len());
    for (got, want) in sol.t.iter().zip(t_eval.iter()) {
        assert!((got - want).abs() <= 1e-12, "got {}, want {}", got, want);
    }
}

#[test]
fn backward_integration_returns_to_initial_state() {
    let q0 = [1.0];
    let v0 = [0.0];
    let sol = Ivp::second_order(
        &HarmonicSecondOrder,
        0.0,
        -2.0 * std::f64::consts::PI,
        &q0,
        &v0,
    )
    .method(SymplecticMethod::VelocityVerlet)
    .step_size(-0.01)
    .solve()
    .expect("Ivp::solve failed");

    assert_eq!(sol.status, Status::Success);
    let y_end = sol.y.last().unwrap();
    assert!((y_end[0] - 1.0).abs() < 5e-4, "q_end={}", y_end[0]);
    assert!(y_end[1].abs() < 5e-4, "p_end={}", y_end[1]);
}

#[test]
fn dense_output_matches_step_endpoints() {
    let q0 = [1.0];
    let p0 = [0.0];
    let sol = Ivp::hamiltonian(&HarmonicHamiltonian, 0.0, 2.0, &q0, &p0)
        .method(SymplecticMethod::Yoshida4)
        .step_size(0.1)
        .dense_output(true)
        .solve()
        .expect("Ivp::solve failed");

    let ys_dense = sol.sol_many(&sol.t).expect("dense evaluation failed");
    assert_eq!(ys_dense.len(), sol.y.len());
    for (dense, stored) in ys_dense.iter().zip(sol.y.iter()) {
        for (a, b) in dense.iter().zip(stored.iter()) {
            assert!((a - b).abs() <= 1e-10, "{} != {}", a, b);
        }
    }
}

#[test]
fn dense_output_interpolates_midpoint_reasonably() {
    let q0 = [1.0];
    let p0 = [0.0];
    let sol = Ivp::hamiltonian(&HarmonicHamiltonian, 0.0, 1.0, &q0, &p0)
        .method(SymplecticMethod::VelocityVerlet)
        .step_size(0.05)
        .dense_output(true)
        .solve()
        .expect("Ivp::solve failed");

    let y_mid = sol.sol(0.5).expect("dense midpoint evaluation failed");
    assert!((y_mid[0] - 0.5f64.cos()).abs() < 2e-3, "q_mid={}", y_mid[0]);
    assert!((y_mid[1] + 0.5f64.sin()).abs() < 2e-3, "p_mid={}", y_mid[1]);
}

#[test]
fn max_steps_returns_partial_solution() {
    let q0 = [1.0];
    let p0 = [0.0];
    let sol = Ivp::hamiltonian(&HarmonicHamiltonian, 0.0, 1.0, &q0, &p0)
        .method(SymplecticMethod::VelocityVerlet)
        .step_size(0.1)
        .max_steps(3usize)
        .solve()
        .expect("Ivp::solve failed");

    assert_eq!(sol.status, Status::NeedLargerNMax);
    assert_eq!(sol.nstep, 3);
    assert!(sol.t.last().copied().unwrap() < 1.0);
}

#[test]
fn mismatched_dimensions_error() {
    let q0 = [1.0, 2.0];
    let p0 = [0.0];
    let err = Ivp::hamiltonian(&HarmonicHamiltonian, 0.0, 1.0, &q0, &p0)
        .method(SymplecticMethod::VelocityVerlet)
        .step_size(0.1)
        .solve()
        .expect_err("expected dimension mismatch");

    match err {
        Error::Config(ConfigError::DimensionMismatch { .. }) => {}
        other => panic!("unexpected error: {:?}", other),
    }
}
