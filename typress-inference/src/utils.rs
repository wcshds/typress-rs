use std::path::Path;

use burn::{
    module::Module,
    prelude::Backend,
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::Tensor,
};
use typress_core::{
    image_processing::{ImageReader, NormalizeInfo, SizeInfo},
    model::{
        deit::deit_model::{DeiTModel, DeiTModelConfig},
        trocr::decoder::{TrOCRForCausalLM, TrOCRForCausalLMConfig},
    },
};

pub fn load_deit_model<B: Backend>(device: &B::Device) -> DeiTModel<B> {
    let bfr = BinFileRecorder::<FullPrecisionSettings>::new();
    let deit_model = DeiTModelConfig::new().init::<B>(device);
    let deit_model = deit_model
        .load_file("../weights/deit_model.bin", &bfr, device)
        .unwrap();

    deit_model
}

pub fn load_decoder<B: Backend>(device: &B::Device) -> TrOCRForCausalLM<B> {
    let bfr = BinFileRecorder::<FullPrecisionSettings>::new();
    let decoder = TrOCRForCausalLMConfig::new().init(device);
    let decoder = decoder
        .load_file("../weights/decoder.bin", &bfr, device)
        .unwrap();

    decoder
}

pub fn init_image_tensor<B: Backend, P: AsRef<Path>>(
    paths: &[P],
    device: &B::Device,
) -> Tensor<B, 4> {
    let image = ImageReader::read_images(paths, Some(SizeInfo::new(384, 384)));
    let tensor = image.to_tensor(
        device,
        Some(1.0 / 255.0),
        Some(NormalizeInfo::new([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])),
    );

    tensor
}
