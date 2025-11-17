if ! command -v ptxas &> /dev/null; then
    echo "CUDA not found! Please install the CUDA Toolkit (with, e.g., sudo apt install nvidia-cuda-toolkit)."
    exit 1
fi

if ! command -v mlir-opt &> /dev/null; then
    test -x /home/linuxbrew/.linuxbrew/bin/brew || /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    if ! command -v brew &> /dev/null; then
        echo 'eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"' >> ~/.bashrc
        eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
    fi
    test -x /home/linuxbrew/.linuxbrew/bin/mlir-opt || /home/linuxbrew/.linuxbrew/bin/brew install llvm@20
    if ! command -v mlir-opt &> /dev/null; then
        export PATH=`brew --prefix llvm@20`/bin:$PATH
        export LD_LIBRARY_PATH=`brew --prefix`/lib:`brew --prefix llvm@20`/bin:$LD_LIBRARY_PATH
    fi
fi
