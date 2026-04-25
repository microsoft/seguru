use gpu_host::SafeGpuConfig;

use super::*;

pub(crate) struct Empty {
    bdim_x: usize,
    bdim_y: usize,
    bdim_z: usize,
    gdim_x: usize,
    gdim_y: usize,
    gdim_z: usize,
    shared_size: usize,
}

#[gpu::cuda_kernel]
fn empty() {
    if gpu::thread_id::<gpu::DimX>() >= 1 {
        return;
    }
    gpu::sync::sync_threads();
}

impl Empty {
    pub fn new_from_runner<'a, T: KernelRunner<'a>>(runner: &T) -> Option<Self> {
        let launch_config = runner.launch_config();
        Some(Self {
            bdim_x: launch_config.block_dim_x() as usize,
            bdim_y: launch_config.block_dim_y() as usize,
            bdim_z: launch_config.block_dim_z() as usize,
            gdim_x: launch_config.grid_dim_x() as usize,
            gdim_y: launch_config.grid_dim_y() as usize,
            gdim_z: launch_config.grid_dim_z() as usize,
            shared_size: launch_config.shared_size() as usize,
        })
    }
}

impl<'a> KernelRunner<'a> for Empty {
    fn new<N: gpu_host::GpuCtxSpace>(
        _ctx: &'a gpu_host::GpuCtxGuard<N>,
        config: Config,
    ) -> Option<Self> {
        Some(Self {
            bdim_x: 256,
            bdim_y: 1,
            bdim_z: 1,
            gdim_x: (config.batch_size * config.seq_len * config.channel).div_ceil(256),
            gdim_y: 1,
            gdim_z: 1,
            shared_size: 0,
        })
    }

    fn launch_config(&self) -> impl gpu_host::SafeGpuConfig {
        gpu_host::gpu_config!(
            self.gdim_x as u32,
            self.gdim_y as u32,
            self.gdim_z as u32,
            self.bdim_x as u32,
            self.bdim_y as u32,
            self.bdim_z as u32,
            self.shared_size as u32
        )
    }

    fn rs_fn<N: gpu_host::GpuCtxSpace>(
        &mut self,
        ctx: &gpu_host::GpuCtxGuard<N>,
        m: &gpu_host::GpuModule<N>,
    ) {
        let launch_config = self.launch_config();
        empty::launch(launch_config, ctx, m).expect("kernel launch failed");
    }

    fn c_fn(&mut self) {
        unsafe {
            llmc::empty_host(
                self.gdim_x as _,
                self.gdim_y as _,
                self.gdim_z as _,
                self.bdim_x as _,
                self.bdim_y as _,
                self.bdim_z as _,
                self.shared_size as _,
            );
        }
    }
}
