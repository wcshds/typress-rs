use std::{collections::HashMap, fs};

use serde_json::Value;

const FILE_PATH: &str = "../tokenizer.json";

fn main() {
    println!("cargo::rerun-if-changed={FILE_PATH}");

    let json_str = fs::read_to_string(FILE_PATH).expect(&format!("Cannot find file {FILE_PATH}."));
    let res: Value =
        serde_json::from_str(&json_str).expect(&format!("Cannot parse file {FILE_PATH}."));
    let vocab_value = res["model"]["vocab"].clone();
    let vocab: HashMap<String, u32> = serde_json::from_value(vocab_value)
        .expect("Json file does not contain `[model][vocab]` field or this field is not a map.");
    let vocab_reverse: Vec<(u32, String)> =
        vocab.into_iter().map(|(key, value)| (value, key)).collect();

    fs::write(
        "./src/vocab.rs",
        format!("const VOCAB_VEC: &[(u32, &str)] = &{:?};", vocab_reverse),
    )
    .expect("Failed to write file.");
}
