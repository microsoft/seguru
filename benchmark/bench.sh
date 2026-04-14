# Get rust version number
VER=$(rustc --version | awk '{print $2}')
CRITERION_HOME=results/$VER DISABLE_GPU_BOUND_CHECK=false cargo bench --features llvm
CRITERION_HOME=results/$VER DISABLE_GPU_BOUND_CHECK=true cargo bench --features seguru;
CRITERION_HOME=results/$VER DISABLE_GPU_BOUND_CHECK=false cargo bench --features nvvm --features seguru;