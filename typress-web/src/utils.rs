use burn::{
    module::Module,
    prelude::Backend,
    record::{BinBytesRecorder, FullPrecisionSettings, Recorder},
};
// use tokenizers::Tokenizer;
use typress_core::model::{
    deit::deit_model::{DeiTModel, DeiTModelConfig},
    trocr::decoder::{TrOCRForCausalLM, TrOCRForCausalLMConfig},
};

static DEIT_MODEL_BIN: &[u8] = include_bytes!("../../weights/deit_model.bin");
static DECODER_BIN: &[u8] = include_bytes!("../../weights/decoder.bin");
// static TOKENIZER_JSON: &[u8] = include_bytes!("../../tokenizer.json");

pub fn load_deit_model<B: Backend>(device: &B::Device) -> DeiTModel<B> {
    let bbr = BinBytesRecorder::<FullPrecisionSettings>::new();

    let deit_model = DeiTModelConfig::new().init::<B>(device);
    let record = bbr
        .load(DEIT_MODEL_BIN.to_vec(), device)
        .expect("Failed to load deit model");
    let deit_model = deit_model.load_record(record);

    deit_model
}

pub fn load_decoder<B: Backend>(device: &B::Device) -> TrOCRForCausalLM<B> {
    let bbr = BinBytesRecorder::<FullPrecisionSettings>::new();

    let decoder = TrOCRForCausalLMConfig::new().init::<B>(device);
    let record = bbr
        .load(DECODER_BIN.to_vec(), device)
        .expect("Failed to load decoder");
    let decoder = decoder.load_record(record);

    decoder
}
