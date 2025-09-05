//! SciPy-like solve_ivp entry point implementation

use crate::{
    Float,
    core::{ode::ODE, status::Status},
    error::Error,
    methods::{
        dp::{dop853, dopri5},
        rk::{rk4, rk23},
        settings::Settings,
    },
};

use super::{
    options::{IVPOptions, Method},
    solout::DefaultSolOut,
};

/// Rich solution of solve_ivp: sampled data plus basic stats
#[derive(Debug, Clone)]
pub struct IVPSolution {
    pub t: Vec<Float>,
    pub y: Vec<Vec<Float>>,
    pub nfev: usize,
    pub nstep: usize,
    pub naccpt: usize,
    pub nrejct: usize,
    pub status: Status,
}

/// Solve an initial value problem with SciPy-like options.
pub fn solve_ivp<'a, F>(
    f: &F,
    x0: Float,
    xend: Float,
    y0: &[Float],
    options: IVPOptions,
) -> Result<IVPSolution, Vec<Error>>
where
    F: ODE,
{
    // Build Settings (rtol/atol are passed to methods)
    let mut settings = Settings::builder().build();
    if let Some(nmax) = options.nmax {
        settings.nmax = nmax;
    }
    settings.h0 = options.first_step;
    settings.hmax = options.max_step;
    settings.hmin = options.min_step;

    // Prepare the default SolOut (wrapping user callback if provided)
    let mut default_solout = DefaultSolOut::new(options.t_eval);

    // Dispatch by method
    let result = match options.method {
        Method::RK4 => {
            let h = settings.h0.unwrap_or_else(|| (xend - x0) / 100.0);
            rk4(f, x0, xend, y0, h, Some(&mut default_solout), settings)
        }
        Method::RK23 => rk23(
            f,
            x0,
            xend,
            y0,
            options.rtol,
            options.atol,
            Some(&mut default_solout),
            settings,
        ),
        Method::RK45 => dopri5(
            f,
            x0,
            xend,
            y0,
            options.rtol,
            options.atol,
            Some(&mut default_solout),
            settings,
        ),
        Method::DOP853 => dop853(
            f,
            x0,
            xend,
            y0,
            options.rtol,
            options.atol,
            Some(&mut default_solout),
            settings,
        ),
    };

    match result {
        Ok(sol) => {
            let (t, y) = default_solout.into_data();
            Ok(IVPSolution {
                t,
                y,
                nfev: sol.nfev,
                nstep: sol.nstep,
                naccpt: sol.naccpt,
                nrejct: sol.nrejct,
                status: sol.status,
            })
        }
        Err(errors) => {
            return Err(errors);
        }
    }
}
