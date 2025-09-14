//! # Example: Circular Restricted Three-Body Problem (CR3BP)
//!
//! Solve the circular restricted three-body problem (CR3BP) as a first-order system.
//!
//! Equations:
//! dx/dt = vx
//! dy/dt = vy
//! dz/dt = vz
//! dvx/dt = x + 2*vy - (1-mu)*(x+mu)/r13^3 - mu*(x-1+mu)/r23^3
//! dvy/dt = y - 2*vx - (1-mu)*y/r13^3 - mu*y/r23^3
//! dvz/dt = -(1-mu)*z/r13^3 - mu*z/r23^3
//! where r13 and r23 are distances to the primary and secondary bodies.
//!
//! Initial conditions:
//! Mass ratio: mu = 0.1
//! x(0) = 0.5, y(0) = 0, z(0) = 0
//! vx(0) = 0, vy(0) = 1.2, vz(0) = 0
//! t0 = 0, t_end = 10
//!

use ivp::prelude::*;

struct CR3BP {
    mu: f64, // Mass ratio
}

impl ODE for CR3BP {
    fn ode(&self, _t: f64, sv: &[f64], dsdt: &mut [f64]) {
        // Components
        let (x, y, _z, vx, vy, vz) = (sv[0], sv[1], sv[2], sv[3], sv[4], sv[5]);

        // Distances to the primary and secondary bodies
        let r13 = ((x + self.mu).powi(2) + y.powi(2) + _z.powi(2)).sqrt();
        let r23 = ((x - 1.0 + self.mu).powi(2) + y.powi(2) + _z.powi(2)).sqrt();

        dsdt[0] = vx;
        dsdt[1] = vy;
        dsdt[2] = vz;
        dsdt[3] = x + 2.0 * vy
            - (1.0 - self.mu) * (x + self.mu) / r13.powi(3)
            - self.mu * (x - 1.0 + self.mu) / r23.powi(3);
        dsdt[4] = y - 2.0 * vx - (1.0 - self.mu) * y / r13.powi(3) - self.mu * y / r23.powi(3);
        dsdt[5] = -(1.0 - self.mu) * vz / r13.powi(3) - self.mu * vz / r23.powi(3);
    }

    fn event(&self, _x: f64, sv: &[f64], event: &mut EventConfig) -> f64 {
        // Terminate after 1 occurrence of the event
        event.terminal();
        event.positive(); // Only detect positive-going zero crossings
        // Other options: event.negative(), event.all() <- All is default feel free to omit
        // For recording multiple events before termination, use event.terminal_count(n) instead of event.terminal()

        // Example event: crossing the x-axis (sv=0)
        sv[1]
    }
}

fn main() {
    let mu = 0.1; // Mass ratio
    let cr3bp = CR3BP { mu };
    let t0 = 0.0;
    let y0 = [0.5, 0.1, 0.0, 0.0, 1.2, 0.0]; // Initial state vector
    let tf = 10.0; // Integrate over 10 time units
    let t_eval = (0..10).map(|i| i as f64 * 1.0).collect();
    let options = Options::builder()
        .method(Method::DOP853)
        .rtol(1e-6)
        .atol(1e-9)
        .t_eval(t_eval)
        .build();

    match solve_ivp(&cr3bp, t0, tf, &y0, options) {
        Ok(sol) => {
            println!("Final status: {:?}", sol.status);
            println!("Number of function evaluations: {}", sol.nfev);
            println!("Number of steps taken: {}", sol.nstep);
            println!("Number of accepted steps: {}", sol.naccpt);
            println!("Number of rejected steps: {}", sol.nrejct);

            println!("t and y at t_eval:");
            for (ti, yi) in sol.iter() {
                println!("t = {:>8.5}, y = {:>8.5?}", ti, yi);
            }

            println!("t and y at events:");
            for (ti, yi) in sol.events() {
                println!("t = {:>8.5}, y = {:>8.5?}", ti, yi);
            }
        }
        Err(err) => eprintln!("Integration failed: {:?}", err),
    }
}
