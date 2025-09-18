#[derive(Debug)]
#[allow(dead_code)]
pub enum GpuCodegenError {
    MisuseMutableArgument,
    MissingSyncThreads,
    InvalidDiversedData,
    UnsupportedAsm(String),
    InvalidMLIR(String),
}

pub type GpuCodegenResult<T> = Result<T, GpuCodegenError>;

impl GpuCodegenError {
    pub fn fatal(self, tcx: rustc_middle::ty::TyCtxt<'_>) -> ! {
        tcx.sess.dcx().fatal(format!("{:?}", self))
    }
}
