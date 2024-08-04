# Burn implementation of [Typress](https://github.com/ParaN3xus/typress)

## Example
```
git clone https://github.com/wcshds/typress-rs.git
cd typress-rs/typress-inference

// Specify either backend for inference
cargo run --release --features ndarray
cargo run --release --features ndarray-blas-openblas
cargo run --release --features ndarray-blas-netlib
cargo run --release --features tch-gpu
cargo run --release --features tch-cpu
cargo run --release --features candle-gpu
cargo run --release --features candle-cpu
cargo run --release --features wgpu
```