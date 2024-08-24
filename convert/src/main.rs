use std::{fs, process::Command};

use burn::{
    backend::{ndarray::NdArrayDevice, NdArray},
    module::Module,
    record::{BinFileRecorder, FullPrecisionSettings, PrettyJsonFileRecorder},
};
use typress_core::model::{
    deit::deit_model::DeiTModelConfig, trocr::decoder::TrOCRForCausalLMConfig,
};

fn main() {
    println!("Converting the PyTorch model file to Burn format...");

    let device = NdArrayDevice::Cpu;
    let pfr = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();
    let bfr = BinFileRecorder::<FullPrecisionSettings>::new();

    let deit_model = DeiTModelConfig::new().init::<NdArray>(&device);
    let decoder = TrOCRForCausalLMConfig::new().init::<NdArray>(&device);

    deit_model
        .clone()
        .save_file("./deit_model.json", &pfr)
        .unwrap();
    decoder.clone().save_file("./decoder.json", &pfr).unwrap();

    read_and_write_parameters();

    let deit_model = deit_model
        .load_file("./deit_model.json", &pfr, &device)
        .expect("Failed to read converted model.");
    let decoder = decoder
        .load_file("./decoder.json", &pfr, &device)
        .expect("Failed to read converted model.");

    deit_model
        .save_file("../weights/deit_model.bin", &bfr)
        .expect("Failed to write the converted model.");
    decoder
        .save_file("../weights/decoder.bin", &bfr)
        .expect("Failed to write the converted model.");

    fs::remove_file("./deit_model.json").expect("Failed to delete intermediate files.");
    fs::remove_file("./decoder.json").expect("Failed to delete intermediate files.");

    println!("Converting the file successfully.");
}

fn read_and_write_parameters() {
    let output = Command::new("python")
        .arg("./convert.py")
        .output()
        .expect("Failed to execute python script.");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("Failed to read and write parameters: {}", stderr);
    }
}
