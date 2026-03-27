if ! command -v ptxas &> /dev/null; then
    echo "CUDA not found! Please install the CUDA Toolkit (with, e.g., sudo apt install nvidia-cuda-toolkit)."
    exit 1
fi

if ! command -v mlir-opt &> /dev/null; then
    wget https://apt.llvm.org/llvm.sh
    chmod +x llvm.sh
    sudo ./llvm.sh 20
    sudo apt-get install libmlir-20-dev mlir-20-tools libpolly-20-dev
    export PATH=/usr/lib/llvm-20/bin:$PATH
    export LD_LIBRARY_PATH=/usr/lib/llvm-20/lib:$LD_LIBRARY_PATH
fi
