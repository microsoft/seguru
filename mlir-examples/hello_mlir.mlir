module attributes {gpu.container_module} {
    gpu.module @gpu  attributes {visibility = "public"} {
        gpu.func @kernel_print() kernel  {
            %0 = gpu.thread_id x
            %csti8 = arith.constant 2 : i8
            %cstf32 = arith.constant 3.0 : f32
            gpu.printf "Hello from %lld, %d, %f\n", %0, %csti8, %cstf32  : index, i8, f32
            gpu.return
        }
    }

}
