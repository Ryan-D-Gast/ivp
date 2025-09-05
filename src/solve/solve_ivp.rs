//! SciPy-like solve_ivp entry point implementation

use crate::{
    Float,
    error::Error,
    methods::{
        dp::{dop853, dopri5},
        rk::{rk4, rk23},
        settings::Settings,
    },
    prelude::{ODE, SolOut, Status},
};

use super::{options::{IVPOptions, Method}, solout::DefaultSolOut};

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
pub fn solve_ivp<F, S>(
    f: &F,
    x0: Float,
    xend: Float,
    y0: &[Float],
    mut options: IVPOptions<'_, S>,
) -> Result<IVPSolution, Error>
where
    F: ODE,
    S: SolOut,
{
    // Build Settings (rtol/atol are passed to methods)
    let mut settings = Settings::builder().build();
    if let Some(nmax) = options.nmax {
        settings.nmax = nmax;
    }
    settings.h0 = options.first_step;
    settings.hmax = options.max_step;
    settings.hmin = options.min_step;

    // Validate t_eval if provided
    if let Some(ref te) = options.t_eval {
        if te.is_empty() {
            return Err(Error::InvalidTEval(
                "t_eval must be non-empty when provided".into(),
            ));
        }
        let dir = (xend - x0).signum();
        let mut prev = te[0];
        for &t in te.iter().skip(1) {
            if !(dir >= 0.0 && t >= prev || dir <= 0.0 && t <= prev) {
                return Err(Error::InvalidTEval(
                    "t_eval must be monotonic in the integration direction".into(),
                ));
            }
            prev = t;
        }
        let min_t = te.first().copied().unwrap();
        let max_t = te.last().copied().unwrap();
        let (lo, hi) = if dir >= 0.0 { (x0, xend) } else { (xend, x0) };
        if min_t < lo - 1e-12 || max_t > hi + 1e-12 {
            return Err(Error::InvalidTEval(
                "t_eval points must lie within [x0, xend]".into(),
            ));
        }
    }

    // Prepare the default SolOut (wrapping user callback if provided)
    let mut default_solout = DefaultSolOut::new(
        options.t_eval.as_deref(),
        options.save_step_endpoints,
        options.solout.as_deref_mut(),
    );

    // Dispatch by method
    match options.method {
        Method::RK4 => {
            let h = settings.h0.unwrap_or_else(|| (xend - x0) / 100.0);
            let res = rk4(f, x0, xend, y0, h, Some(&mut default_solout), settings);
            res.map(|solution| {
                let (t, y) = default_solout.into_data();
                IVPSolution {
                    t,
                    y,
                    nfev: solution.nfev,
                    nstep: solution.nstep,
                    naccpt: solution.naccpt,
                    nrejct: solution.nrejct,
                    status: solution.status,
                }
            })
        }
        Method::RK23 => rk23(
            f,
            x0,
            xend,
            y0,
            options.rtol.clone(),
            options.atol.clone(),
            Some(&mut default_solout),
            settings,
        )
        .map_err(Error::MethodErrors)
        .map(|solution| {
            let (t, y) = default_solout.into_data();
            IVPSolution {
                t,
                y,
                nfev: solution.nfev,
                nstep: solution.nstep,
                naccpt: solution.naccpt,
                nrejct: solution.nrejct,
                status: solution.status,
            }
        }),
        Method::RK45 => dopri5(
            f,
            x0,
            xend,
            y0,
            options.rtol.clone(),
            options.atol.clone(),
            Some(&mut default_solout),
            settings,
        )
        .map_err(Error::MethodErrors)
        .map(|solution| {
            let (t, y) = default_solout.into_data();
            IVPSolution {
                t,
                y,
                nfev: solution.nfev,
                nstep: solution.nstep,
                naccpt: solution.naccpt,
                nrejct: solution.nrejct,
                status: solution.status,
            }
        }),
        Method::DOP853 => dop853(
            f,
            x0,
            xend,
            y0,
            options.rtol,
            options.atol,
            Some(&mut default_solout),
            settings,
        )
        .map_err(Error::MethodErrors)
        .map(|solution| {
            let (t, y) = default_solout.into_data();
            IVPSolution {
                t,
                y,
                nfev: solution.nfev,
                nstep: solution.nstep,
                naccpt: solution.naccpt,
                nrejct: solution.nrejct,
                status: solution.status,
            }
        }),
    }
}
