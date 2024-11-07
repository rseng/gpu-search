# https://github.com/mrvollger/fibertools-rs

```console
README.md:However, due to size constraints in `bioconda` this version does not support contain the pytorch libraries or GPU acceleration for m6A predictions. m6A predictions will still work in the bioconda version but may be much slower. If you would like to use m6A prediction and GPU acceleration, you will need to install using the directions in the fibertools [book](https://fiberseq.github.io/fibertools/install.html).
Cargo.toml:burn = { version = "0.12", optional = true, features = ["candle"] } # "wgpu",
Cargo.lock: "burn-wgpu",
Cargo.lock:name = "burn-wgpu"
Cargo.lock: "wgpu",
Cargo.lock:name = "gpu-alloc"
Cargo.lock: "gpu-alloc-types",
Cargo.lock:name = "gpu-alloc-types"
Cargo.lock:name = "gpu-allocator"
Cargo.lock:name = "gpu-descriptor"
Cargo.lock: "gpu-descriptor-types",
Cargo.lock:name = "gpu-descriptor-types"
Cargo.lock:name = "wgpu"
Cargo.lock: "wgpu-core",
Cargo.lock: "wgpu-hal",
Cargo.lock: "wgpu-types",
Cargo.lock:name = "wgpu-core"
Cargo.lock: "wgpu-hal",
Cargo.lock: "wgpu-types",
Cargo.lock:name = "wgpu-hal"
Cargo.lock: "gpu-alloc",
Cargo.lock: "gpu-allocator",
Cargo.lock: "gpu-descriptor",
Cargo.lock: "wgpu-types",
Cargo.lock:name = "wgpu-types"
src/cli/predict_opts.rs:    /// Increasing improves GPU performance at the cost of memory.
src/m6a_burn/mod.rs:        let device = if tch::utils::has_cuda() {
src/m6a_burn/mod.rs:            LibTorchDevice::Cuda(0)
src/subcommands/predict_m6a.rs:    /// group reads together for predictions so we have to move data to the GPU less often

```
