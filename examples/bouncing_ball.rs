//! Bouncing ball with event detection for ground impact.

use ivp::prelude::*;

struct BouncingBall {
    gravity: f64,
    drag: f64,
}

impl IVP for BouncingBall {
    fn ode(&self, _t: f64, state: &[f64], dsdt: &mut [f64]) {
        let vy = state[1];
        dsdt[0] = vy;
        dsdt[1] = -self.gravity - self.drag * vy * vy.abs();
    }

    fn events(&self, _t: f64, state: &[f64], out: &mut [f64]) {
        out[0] = state[0]; // Ground impact when y = 0
    }

    fn n_events(&self) -> usize {
        1
    }

    fn event_config(&self, _index: usize) -> EventConfig {
        let mut config = EventConfig::default();
        config.terminal();
        config.negative();
        config
    }
}

fn main() {
    let ball = BouncingBall { gravity: 9.81, drag: 0.02 };
    let y0 = [10.0, 5.0]; // [height, velocity]

    let options = Options::builder()
        .method(Method::DOPRI5)
        .rtol(1e-8)
        .atol(1e-10)
        .build();

    match solve_ivp(&ball, 0.0, 10.0, &y0, options) {
        Ok(sol) => {
            println!("Status: {:?}", sol.status);

            if let Some(t_impact) = sol.t_events.get(0).and_then(|e| e.first()) {
                let v_impact = sol.y_events[0][0][1].abs();
                println!("Ground impact at t={:.4}s, velocity={:.4} m/s", t_impact, v_impact);
            }

            println!("\nTrajectory:");
            for (t, y) in sol.iter().take(10) {
                println!("t={:.4}: height={:.4}, velocity={:.4}", t, y[0], y[1]);
            }
        }
        Err(e) => eprintln!("Error: {:?}", e),
    }
}
