#[cfg(any(
    feature = "ndarray",
    feature = "ndarray-blas-netlib",
    feature = "ndarray-blas-openblas",
    feature = "ndarray-blas-accelerate",
))]
mod ndarray {
    use burn::backend::ndarray::{NdArray, NdArrayDevice};
    use typress_inference::inference;

    pub fn run() {
        let device = NdArrayDevice::Cpu;
        inference::<NdArray>(&device);
    }
}

#[cfg(feature = "tch-gpu")]
mod tch_gpu {
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};
    use typress_inference::inference;

    pub fn run() {
        let device = LibTorchDevice::Cuda(0);
        inference::<LibTorch>(&device);
    }
}

#[cfg(feature = "tch-cpu")]
mod tch_cpu {
    use burn::backend::libtorch::{LibTorch, LibTorchDevice};
    use typress_inference::inference;

    pub fn run() {
        let device = LibTorchDevice::Cpu;
        inference::<LibTorch>(&device);
    }
}

#[cfg(feature = "candle-gpu")]
mod candle_gpu {
    use burn::backend::candle::{Candle, CandleDevice};
    use typress_inference::inference;

    pub fn run() {
        let device = CandleDevice::Cuda(0);
        inference::<Candle>(&device);
    }
}

#[cfg(feature = "candle-cpu")]
mod candle_cpu {
    use burn::backend::candle::{Candle, CandleDevice};
    use typress_inference::inference;

    pub fn run() {
        let device = CandleDevice::Cpu;
        inference::<Candle>(&device);
    }
}

#[cfg(feature = "wgpu")]
mod wgpu {
    use burn::backend::wgpu::{Wgpu, WgpuDevice};
    use typress_inference::inference;

    pub fn run() {
        let device = WgpuDevice::Cpu;
        inference::<Wgpu>(&device);
    }
}

fn main() {
    #[cfg(any(
        feature = "ndarray",
        feature = "ndarray-blas-netlib",
        feature = "ndarray-blas-openblas",
        feature = "ndarray-blas-accelerate",
    ))]
    ndarray::run();
    #[cfg(feature = "tch-gpu")]
    tch_gpu::run();
    #[cfg(feature = "tch-cpu")]
    tch_cpu::run();
    #[cfg(feature = "candle-gpu")]
    candle_gpu::run();
    #[cfg(feature = "candle-cpu")]
    candle_cpu::run();
    #[cfg(feature = "wgpu")]
    wgpu::run();
}
