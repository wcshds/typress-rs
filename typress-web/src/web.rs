#![allow(clippy::new_without_default)]

use alloc::vec::Vec;
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

/// The image classifier
#[wasm_bindgen]
pub struct TrOCR {
    model: ModelType,
}

#[wasm_bindgen]
impl TrOCR {
    /// Constructor called by JavaScripts with the new keyword.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        log::info!("Initializing the image classifier");
        let device = Default::default();
        Self {
            model: ModelType::WithNdArrayBackend(Model::new(&device)),
        }
    }

    /// Runs inference on the image
    pub async fn inference(&self, input: &[f32]) -> Result<JsValue, JsValue> {
        log::info!("Generate Typst formula from the image");

        // let tokenizer = load_tokenizer();
        let start = Instant::now();

        let res_ids = match self.model {
            ModelType::WithCandleBackend(ref model) => model.generate(input).await,
            ModelType::WithNdArrayBackend(ref model) => model.generate(input).await,
            ModelType::WithWgpuBackend(ref model) => model.generate(input).await,
        };
        let res_str = decode(&res_ids, true);

        let duration = start.elapsed();

        log::debug!("Inference is completed in {:?}", duration);

        Ok(serde_wasm_bindgen::to_value(&res_ids)?)
    }

    /// Sets the backend to Candle
    pub async fn set_backend_candle(&mut self) -> Result<(), JsValue> {
        log::info!("Loading the model to the Candle backend");
        let start = Instant::now();
        let device = Default::default();
        self.model = ModelType::WithCandleBackend(Model::new(&device));
        let duration = start.elapsed();
        log::debug!("Model is loaded to the Candle backend in {:?}", duration);
        Ok(())
    }

    /// Sets the backend to NdArray
    pub async fn set_backend_ndarray(&mut self) -> Result<(), JsValue> {
        log::info!("Loading the model to the NdArray backend");
        let start = Instant::now();
        let device = Default::default();
        self.model = ModelType::WithNdArrayBackend(Model::new(&device));
        let duration = start.elapsed();
        log::debug!("Model is loaded to the NdArray backend in {:?}", duration);
        Ok(())
    }

    /// Sets the backend to Wgpu
    pub async fn set_backend_wgpu(&mut self) -> Result<(), JsValue> {
        log::info!("Loading the model to the Wgpu backend");
        let start = Instant::now();
        let device = WgpuDevice::default();
        init_async::<AutoGraphicsApi>(&device, Default::default()).await;
        self.model = ModelType::WithWgpuBackend(Model::new(&device));
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

#[allow(clippy::large_enum_variant)]
/// The model is loaded to a specific backend
pub enum ModelType {
    /// The model is loaded to the Candle backend
    WithCandleBackend(Model<Candle<f32, i64>>),

    /// The model is loaded to the NdArray backend
    WithNdArrayBackend(Model<NdArray<f32>>),

    /// The model is loaded to the Wgpu backend
    WithWgpuBackend(Model<Wgpu<f32, i32>>),
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

    /// generate labels for each id
    pub async fn generate(&self, input: &[f32]) -> Vec<u32> {
        let input = Tensor::<B, 1>::from_floats(input, &B::Device::default())
            .reshape([1, CHANNELS, HEIGHT, WIDTH]);

        let encoder_res = self.encoder.forward(input);
        let res_ids = self.decoder.generate(encoder_res, 200, true);

        // batch_size is 1, so it is safe to ignore the first dimension
        let tensor_data = res_ids.into_data().convert::<i64>();
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
