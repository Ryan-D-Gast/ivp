//! Arenstorf orbit in the Circular Restricted Three-Body Problem (CR3BP).
//! 
//! This is a famous periodic orbit discovered by Richard Arenstorf, used as a
//! benchmark problem in numerical ODE literature (Hairer, Norsett & Wanner).
//! The orbit represents a spacecraft's path in the Earth-Moon system.

use ivp::prelude::*;

struct CR3BP {
    mu: f64,
}

impl CR3BP {
    fn jacobi_constant(&self, state: &[f64]) -> f64 {
        let (x, y, z, vx, vy, vz) = (state[0], state[1], state[2], state[3], state[4], state[5]);
        let r1 = ((x + self.mu).powi(2) + y.powi(2) + z.powi(2)).sqrt();
        let r2 = ((x - 1.0 + self.mu).powi(2) + y.powi(2) + z.powi(2)).sqrt();
        let u = 0.5 * (x.powi(2) + y.powi(2)) + (1.0 - self.mu) / r1 + self.mu / r2;
        2.0 * u - (vx.powi(2) + vy.powi(2) + vz.powi(2))
    }
}

impl IVP for CR3BP {
    fn ode(&self, _t: f64, sv: &[f64], dsdt: &mut [f64]) {
        let (x, y, z, vx, vy, vz) = (sv[0], sv[1], sv[2], sv[3], sv[4], sv[5]);
        let r1 = ((x + self.mu).powi(2) + y.powi(2) + z.powi(2)).sqrt();
        let r2 = ((x - 1.0 + self.mu).powi(2) + y.powi(2) + z.powi(2)).sqrt();

        dsdt[0] = vx;
        dsdt[1] = vy;
        dsdt[2] = vz;
        dsdt[3] = x + 2.0 * vy - (1.0 - self.mu) * (x + self.mu) / r1.powi(3) - self.mu * (x - 1.0 + self.mu) / r2.powi(3);
        dsdt[4] = y - 2.0 * vx - (1.0 - self.mu) * y / r1.powi(3) - self.mu * y / r2.powi(3);
        dsdt[5] = -(1.0 - self.mu) * z / r1.powi(3) - self.mu * z / r2.powi(3);
    }
}

fn main() {
    // Earth-Moon mass ratio
    let mu = 0.012277471;
    let cr3bp = CR3BP { mu };

    // Arenstorf orbit initial conditions (periodic orbit, period T ~ 17.0652)
    // From Hairer, Norsett & Wanner "Solving ODEs I"
    let y0 = [
        0.994,       // x
        0.0,         // y
        0.0,         // z
        0.0,         // vx
        -2.00158510637908252240537862224, // vy
        0.0,         // vz
    ];
    let period = 17.0652165601579625588917206249;
    let c_initial = cr3bp.jacobi_constant(&y0);

    let t_eval: Vec<f64> = (0..=100).map(|i| i as f64 * period / 100.0).collect();
    let options = Options::builder()
        .method(Method::DOP853)
        .rtol(1e-12)
        .atol(1e-14)
        .t_eval(t_eval)
        .build();

    match solve_ivp(&cr3bp, 0.0, period, &y0, options) {
        Ok(sol) => {
            println!("Arenstorf Orbit (periodic, T={:.4})", period);
            println!("Status: {:?}, nfev: {}, steps: {}", sol.status, sol.nfev, sol.nstep);

            if let (Some(y_final), Some(&t_final)) = (sol.y.last(), sol.t.last()) {
                let c_final = cr3bp.jacobi_constant(y_final);
                println!("Jacobi constant error: {:.2e}", (c_final - c_initial).abs());
                println!("Position error at T: dx={:.2e}, dy={:.2e}", 
                    (y_final[0] - y0[0]).abs(), (y_final[1] - y0[1]).abs());
                println!("Final t: {:.10}", t_final);
            }

            println!("\nTrajectory (x, y):");
            for (t, y) in sol.iter().step_by(10) {
                println!("t={:6.2}: ({:>9.5}, {:>9.5})", t, y[0], y[1]);
            }
        }
        Err(e) => eprintln!("Error: {:?}", e),
    }
}
