use std::time::Instant;

use burn::{
    backend::{
        libtorch::LibTorchDevice, ndarray::NdArrayDevice, wgpu::WgpuDevice, LibTorch, NdArray, Wgpu,
    },
    module::Module,
    record::{BinFileRecorder, FullPrecisionSettings},
};
use tokenizers::Tokenizer;
use typress_rs::{
    image_processing::{ImageReader, NormalizeInfo, SizeInfo},
    model::{
        deit::deit_model::{DeiTModel, DeiTModelConfig},
        trocr::decoder::{TrOCRForCausalLM, TrOCRForCausalLMConfig},
    },
};

type Backend = LibTorch;
const DEVICE: LibTorchDevice = LibTorchDevice::Cpu;

fn main() {
    let start = Instant::now();

    let encoder = load_deit_model();
    let decoder = load_decoder();
    let tokenizer = Tokenizer::from_file("./tokenizer.json").unwrap();

    let image = ImageReader::read_images(
        &["./images/01.png", "./images/02.png", "./images/03.png"],
        Some(SizeInfo::new(384, 384)),
    );
    let tensor = image.to_tensor(
        &DEVICE,
        Some(1.0 / 255.0),
        Some(NormalizeInfo::new([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])),
    );
    let encoder_res = encoder.forward(tensor);

    let res_ids = decoder.generate(encoder_res, 200, true);

    for each in res_ids.iter_dim(0) {
        let tensor_data = each.into_data().convert::<i64>();
        let tensor_slice = tensor_data.as_slice::<i64>().unwrap();
        let mut tensor_vec = Vec::new();
        for &each in tensor_slice {
            if each == 2 {
                break;
            } else {
                tensor_vec.push(each as u32);
            }
        }

        let res = tokenizer.decode(&tensor_vec, true).unwrap();
        println!("$ {} $\n", res);
    }

    println!("execution time: {}", start.elapsed().as_secs_f64());
}

fn load_deit_model() -> DeiTModel<Backend> {
    let bfr = BinFileRecorder::<FullPrecisionSettings>::new();
    let deit_model = DeiTModelConfig::new().init::<Backend>(&DEVICE);
    let deit_model = deit_model
        .load_file("./weights/deit_model.bin", &bfr, &DEVICE)
        .unwrap();

    deit_model
}

fn load_decoder() -> TrOCRForCausalLM<Backend> {
    let bfr = BinFileRecorder::<FullPrecisionSettings>::new();
    let decoder = TrOCRForCausalLMConfig::new().init(&DEVICE);
    let decoder = decoder
        .load_file("./weights/decoder.bin", &bfr, &DEVICE)
        .unwrap();

    decoder
}
