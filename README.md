# Burn implementation of [Typress](https://github.com/ParaN3xus/typress)

## Inference Example
```
git clone https://github.com/wcshds/typress-rs.git
cd typress-rs/typress-inference

// Specify either backend for inference

// ndarray
cargo run --release --features ndarray
cargo run --release --features ndarray-blas-openblas
cargo run --release --features ndarray-blas-netlib

// libtorch
cargo run --release --features tch-gpu
cargo run --release --features tch-cpu

// candle
cargo run --release --features candle-gpu
cargo run --release --features candle-cpu

// wgpu, device is configurable
cargo run --release --features wgpu // default choice is `WgpuDevice::BestAvailable`
cargo run --release --features wgpu -- integrated_gpu // `WgpuDevice::IntegratedGpu(0)`
cargo run --release --features wgpu -- discrete_gpu // `WgpuDevice::DiscreteGpu(0)`
cargo run --release --features wgpu -- cpu // `WgpuDevice::Cpu`
```

## Web Page Example
This example shows how to run the model totally in browser.

Executes following commands in order.

```
cd typress-rs/typress-web

export RUSTFLAGS="-C embed-bitcode=yes -C codegen-units=1 -C opt-level=3 -Ctarget-feature=+simd128 --cfg web_sys_unstable_apis" & wasm-pack build --release --out-dir ../web-page/packages/typress-pkg/simd --target web

cd ../web-page
pnpm install
pnpm dev
```