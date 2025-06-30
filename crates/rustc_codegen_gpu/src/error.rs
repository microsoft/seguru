#[derive(Debug)]
pub enum GpuCodegenError {
    MisuseMutableArgument,
}

pub type GpuCodegenResult<T> = Result<T, GpuCodegenError>;

impl GpuCodegenError {
    pub fn fatal(self, tcx: rustc_middle::ty::TyCtxt<'_>) -> ! {
        tcx.sess.dcx().fatal(format!("{:?}", self))
    }
}
