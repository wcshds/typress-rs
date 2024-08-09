#![allow(clippy::new_without_default)]

use alloc::{string::String, vec::Vec};
use core::convert::Into;
use typress_core::model::{deit::deit_model::DeiTModel, trocr::decoder::TrOCRForCausalLM};

use burn::{
    backend::{
        wgpu::{init_async, AutoGraphicsApi, WgpuDevice},
        Candle, NdArray, Wgpu,
    },
    prelude::*,
};

use wasm_bindgen::prelude::*;
use wasm_timer::Instant;

use crate::utils::{decode, load_decoder, load_deit_model};

#[wasm_bindgen(start)]
pub fn start() {
    // Initialize the logger so that the logs are printed to the console
    console_error_panic_hook::set_once();
    wasm_logger::init(wasm_logger::Config::default());
}

/// The image is 384x384 pixels with 3 channels (RGB)
const HEIGHT: usize = 384;
const WIDTH: usize = 384;
const CHANNELS: usize = 3;

enum ModelBackend {
    NdArray,
    Candle,
    Wgpu,
}

/// The TrOCR text recoginzer
#[wasm_bindgen]
pub struct TrOCR {
    with_ndarray_backend: Option<Model<NdArray<f32>>>,
    with_candle_backend: Option<Model<Candle<f32, i64>>>,
    with_wgpu_backend: Option<Model<Wgpu<f32, i32>>>,
    current_backend: ModelBackend,
}

#[wasm_bindgen]
impl TrOCR {
    /// Constructor called by JavaScripts with the new keyword.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        log::info!("Initializing the Typress model...");
        let device = Default::default();
        Self {
            with_ndarray_backend: Some(Model::new(&device)),
            with_candle_backend: None,
            with_wgpu_backend: None,
            current_backend: ModelBackend::NdArray,
        }
    }

    /// Runs inference on the image
    ///
    /// ## Parameters:
    ///
    /// input: a `[3, height, width]` flattened f32 array, all numbers should be rescaled (* 1./255.)
    pub async fn inference(&self, input: &[f32]) -> Result<String, JsValue> {
        log::info!("Generate Typst formula from the image...");

        // let tokenizer = load_tokenizer();
        let start = Instant::now();

        let res_ids = match self.current_backend {
            ModelBackend::NdArray => {
                let model = self
                    .with_ndarray_backend
                    .as_ref()
                    .expect("Internel error: fail to select NdArray backend.");
                model.generate(input).await
            }
            ModelBackend::Candle => {
                let model = self
                    .with_candle_backend
                    .as_ref()
                    .expect("Internel error: fail to select Candle backend.");
                model.generate(input).await
            }
            ModelBackend::Wgpu => {
                let model = self
                    .with_wgpu_backend
                    .as_ref()
                    .expect("Internel error: fail to select Wgpu backend.");
                model.generate(input).await
            }
        };
        let res_str = decode(&res_ids, true);

        let duration = start.elapsed();

        log::debug!("Inference is completed in {:?}", duration);

        Ok(res_str)
    }

    /// Sets the backend to Candle
    pub async fn set_backend_candle(&mut self) -> Result<(), JsValue> {
        log::info!("Loading the model to the Candle backend");
        let start = Instant::now();
        if self.with_candle_backend.is_none() {
            let device = Default::default();
            self.with_candle_backend = Some(Model::new(&device));
        }
        self.current_backend = ModelBackend::Candle;
        let duration = start.elapsed();
        log::debug!("Model is loaded to the Candle backend in {:?}", duration);
        Ok(())
    }

    /// Sets the backend to NdArray
    pub async fn set_backend_ndarray(&mut self) -> Result<(), JsValue> {
        log::info!("Loading the model to the NdArray backend");
        let start = Instant::now();
        if self.with_ndarray_backend.is_none() {
            let device = Default::default();
            self.with_ndarray_backend = Some(Model::new(&device));
        }
        self.current_backend = ModelBackend::NdArray;
        let duration = start.elapsed();
        log::debug!("Model is loaded to the NdArray backend in {:?}", duration);
        Ok(())
    }

    /// Sets the backend to Wgpu
    pub async fn set_backend_wgpu(&mut self) -> Result<(), JsValue> {
        log::info!("Loading the model to the Wgpu backend");
        let start = Instant::now();
        if self.with_wgpu_backend.is_none() {
            let device = WgpuDevice::default();
            init_async::<AutoGraphicsApi>(&device, Default::default()).await;
            self.with_wgpu_backend = Some(Model::new(&device));
        }
        self.current_backend = ModelBackend::Wgpu;
        let duration = start.elapsed();
        log::debug!("Model is loaded to the Wgpu backend in {:?}", duration);

        log::debug!("Warming up the model");
        let start = Instant::now();
        let _ = self.inference(&[0.0; HEIGHT * WIDTH * CHANNELS]).await;
        let duration = start.elapsed();
        log::debug!("Warming up is completed in {:?}", duration);
        Ok(())
    }
}

/// TrOCR model
pub struct Model<B: Backend> {
    encoder: DeiTModel<B>,
    decoder: TrOCRForCausalLM<B>,
}

impl<B: Backend> Model<B> {
    /// Constructor
    pub fn new(device: &B::Device) -> Self {
        Self {
            encoder: load_deit_model(device),
            decoder: load_decoder(device),
        }
    }

    /// Generate labels for each id.
    ///
    /// ## Parameters:
    ///
    /// input: a `[3, height, width]` flattened f32 array, all numbers should be rescaled (* 1./255.)
    pub async fn generate(&self, input: &[f32]) -> Vec<u32> {
        let device = B::Device::default();
        let input =
            Tensor::<B, 1>::from_floats(input, &device).reshape([1, CHANNELS, HEIGHT, WIDTH]);
        // normalize
        let input = (input
            - Tensor::<B, 1>::from_floats([0.5, 0.5, 0.5], &device).reshape([1, 3, 1, 1]))
            / Tensor::<B, 1>::from_floats([0.5, 0.5, 0.5], &device).reshape([1, 3, 1, 1]);

        let encoder_res = self.encoder.forward(input);
        let res_ids = self.decoder.generate_async(encoder_res, 200, true).await;

        // batch_size is 1, so it is safe to ignore the first dimension
        let tensor_data = res_ids.into_data_async().await.convert::<i64>(); // must be async
        let tensor_slice = tensor_data.as_slice::<i64>().unwrap();
        let mut tensor_vec = Vec::new();

        for &each in tensor_slice {
            if each == 2 {
                break;
            } else {
                tensor_vec.push(each as u32);
            }
        }

        tensor_vec
    }
}
