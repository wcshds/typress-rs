use std::time::Instant;

use burn::prelude::Backend;
use tokenizers::Tokenizer;
use utils::{init_image_tensor, load_decoder, load_deit_model};

pub mod utils;

pub fn inference<B: Backend>(device: &B::Device) {
    let start = Instant::now();

    let encoder = load_deit_model::<B>(device);
    let decoder = load_decoder::<B>(device);
    let tokenizer = Tokenizer::from_file("../tokenizer.json").unwrap();

    let tensor = init_image_tensor(
        &["../images/01.png", "../images/02.png", "../images/03.png"],
        device,
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
