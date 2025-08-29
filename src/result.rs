//! Data structure for holding the result of integrations

use crate::{
    Float,
    status::Status
};

#[derive(Debug, Clone)]
pub struct DPResult<const N: usize> {
    pub x: Float,
    pub y: [Float; N],
    pub h: Float,
    pub status: Status,
    pub nfcns: usize,
    pub nstep: usize,
    pub naccpt: usize,
    pub nrejct: usize,
}