use std::time::Instant;

use burn::{
    backend::{libtorch::LibTorchDevice, ndarray::NdArrayDevice, LibTorch, NdArray},
    module::Module,
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::{Int, Tensor},
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
    let batch_size = encoder_res.dims()[0];

    let decoder_res: Tensor<Backend, 2, Int> = Tensor::zeros([batch_size, 200], &DEVICE);
    let mut decoder_res = decoder_res.slice_assign(
        [0..batch_size, 0..1],
        Tensor::<Backend, 2, Int>::from_ints([[1]], &DEVICE).expand([batch_size as i32, -1]),
    );

    for i in 0..199 {
        let res = decoder.forward(decoder_res.clone(), encoder_res.clone());
        let idx = res
            .slice([0..batch_size, i..(i + 1)])
            .argmax(2)
            .reshape([batch_size, 1]);
        decoder_res = decoder_res.slice_assign([0..batch_size, (i + 1)..(i + 2)], idx.clone());

        if Tensor::all(idx.equal_elem(2))
            .to_data()
            .as_slice::<bool>()
            .unwrap()
            == &[true]
        {
            break;
        };
    }

    let tokenizer = Tokenizer::from_file("./tokenizer.json").unwrap();
    for each in decoder_res.iter_dim(0) {
        let tensor_data = each.into_data();
        let tensor_slice: &[i64] = tensor_data.as_slice::<i64>().unwrap();
        let mut tensor_vec = Vec::with_capacity(200);
        for &each in tensor_slice {
            if each == 2 {
                break;
            } else {
                tensor_vec.push(each as u32);
            }
        }

        let res = tokenizer.decode(&tensor_vec, true).unwrap();
        println!("{}", res);
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
