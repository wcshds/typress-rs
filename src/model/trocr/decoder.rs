use burn::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    prelude::Backend,
    tensor::{Int, Tensor},
};

use super::{
    embed::{
        TrOCRLearnedPositionalEmbedding, TrOCRLearnedPositionalEmbeddingConfig,
        TrOCRScaledWordEmbedding, TrOCRScaledWordEmbeddingConfig,
    },
    layer::{TrOCRDecoderLayer, TrOCRDecoderLayerConfig},
};

#[derive(Module, Debug)]
pub struct TrOCRDecoder<B: Backend> {
    embed_tokens: TrOCRScaledWordEmbedding<B>,
    embed_positions: TrOCRLearnedPositionalEmbedding<B>,
    layernorm_embed: LayerNorm<B>,
    layers: Vec<TrOCRDecoderLayer<B>>,
    dropout: Dropout,
}

impl<B: Backend> TrOCRDecoder<B> {
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        encoder_hidden_states: Tensor<B, 3>,
        past_key_values: Vec<(
            Option<(Tensor<B, 4>, Tensor<B, 4>)>,
            Option<(Tensor<B, 4>, Tensor<B, 4>)>,
        )>,
    ) -> (
        // hidden_states
        Tensor<B, 3>,
        // next_decoder_cache
        Vec<(
            Option<(Tensor<B, 4>, Tensor<B, 4>)>,
            Option<(Tensor<B, 4>, Tensor<B, 4>)>,
        )>,
    ) {
        let [batch_size, target_len] = input_ids.dims();
        let device = encoder_hidden_states.device();
        let input_ids = input_ids.to_device(&device);

        let inputs_embeds = self.embed_tokens.forward(input_ids);

        let past_key_values_lengths = if past_key_values.len() > 0 {
            if let Some(content) = &past_key_values[0].0 {
                content.0.dims()[2]
            } else {
                0
            }
        } else {
            0
        };

        let embed_pos = self
            .embed_positions
            .forward(inputs_embeds.clone(), past_key_values_lengths);
        let hidden_states = inputs_embeds + embed_pos;
        let hidden_states = self.layernorm_embed.forward(hidden_states);
        let mut hidden_states = self.dropout.forward(hidden_states);

        let attention_mask = Tensor::triu(
            Tensor::<B, 2>::ones([target_len, target_len + past_key_values_lengths], &device)
                * (-3.4028234663852886e38),
            1,
        )
        .reshape([1, 1, target_len, target_len + past_key_values_lengths])
        .expand([batch_size as i32, -1, -1, -1]);

        let mut next_decoder_cache = Vec::with_capacity(self.layers.len());
        for (idx, layer) in self.layers.iter().enumerate() {
            let past_key_value = if past_key_values.len() == self.layers.len() {
                (
                    past_key_values[idx].0.clone(),
                    past_key_values[idx].1.clone(),
                )
            } else {
                (None, None)
            };

            let layer_outputs = layer.forward(
                hidden_states,
                Some(attention_mask.clone()),
                Some(encoder_hidden_states.clone()),
                None,
                past_key_value,
            );
            hidden_states = layer_outputs.0;

            next_decoder_cache.push(layer_outputs.1);
        }

        (hidden_states, next_decoder_cache)
    }
}

#[derive(Config)]
pub struct TrOCRDecoderConfig {
    #[config(default = "6")]
    decoder_layers: usize,
    // config.vocab_size
    #[config(default = "1200")]
    vocab_size: usize,
    // config.hidden_size
    #[config(default = "256")]
    embed_dimensions: usize,
    // It is only useful in training, since this project focuses on inference, this parameter do nothing.
    #[config(default = "1")]
    padding_idx: usize,
    #[config(default = "512")]
    max_position_embeddings: usize,
    #[config(default = "8")]
    decoder_attention_heads: usize,
    #[config(default = "384")]
    cross_attention_hidden_size: usize,
    #[config(default = "1024")]
    decoder_ffn_dim: usize,
    #[config(default = "0.0")]
    dropout: f64,
    #[config(default = "0.0")]
    attention_dropout: f64,
    #[config(default = "0.0")]
    activation_dropout: f64,
}

impl TrOCRDecoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TrOCRDecoder<B> {
        let embed_scale = (self.embed_dimensions as f64).sqrt();
        let embed_tokens = TrOCRScaledWordEmbeddingConfig::new()
            .with_num_embed(self.vocab_size)
            .with_embed_dimensions(self.embed_dimensions)
            .with_padding_idx(1)
            .with_embed_scale(embed_scale)
            .init(device);
        let embed_positions = TrOCRLearnedPositionalEmbeddingConfig::new()
            .with_num_embed(self.max_position_embeddings)
            .with_embed_dimensions(self.embed_dimensions)
            .with_offset(2)
            .init(device);
        let layernorm_embed = LayerNormConfig::new(self.embed_dimensions).init(device);
        let layers = (0..self.decoder_layers)
            .map(|_| {
                TrOCRDecoderLayerConfig::new()
                    .with_embed_dimensions(self.embed_dimensions)
                    .with_decoder_attention_heads(self.decoder_attention_heads)
                    .with_cross_attention_hidden_size(self.cross_attention_hidden_size)
                    .with_decoder_ffn_dim(self.decoder_ffn_dim)
                    .with_attention_dropout(self.attention_dropout)
                    .with_activation_dropout(self.activation_dropout)
                    .init(device)
            })
            .collect();
        let dropout = DropoutConfig::new(self.dropout).init();

        TrOCRDecoder {
            embed_tokens,
            embed_positions,
            layernorm_embed,
            layers,
            dropout,
        }
    }
}

#[derive(Module, Debug)]
pub struct TrOCRForCausalLM<B: Backend> {
    model: TrOCRDecoder<B>,
    output_projection: Linear<B>,
}

impl<B: Backend> TrOCRForCausalLM<B> {
    pub fn forward(
        &self,
        input_ids: Tensor<B, 2, Int>,
        encoder_hidden_states: Tensor<B, 3>,
        past_key_values: Vec<(
            Option<(Tensor<B, 4>, Tensor<B, 4>)>,
            Option<(Tensor<B, 4>, Tensor<B, 4>)>,
        )>,
    ) -> (
        // logits
        Tensor<B, 3>,
        // past_key_values
        Vec<(
            Option<(Tensor<B, 4>, Tensor<B, 4>)>,
            Option<(Tensor<B, 4>, Tensor<B, 4>)>,
        )>,
    ) {
        let outputs = self
            .model
            .forward(input_ids, encoder_hidden_states, past_key_values);
        let logits = self.output_projection.forward(outputs.0);

        (logits, outputs.1)
    }
}

#[derive(Config)]
pub struct TrOCRForCausalLMConfig {
    #[config(default = "6")]
    decoder_layers: usize,
    // config.vocab_size
    #[config(default = "1200")]
    vocab_size: usize,
    // config.hidden_size
    #[config(default = "256")]
    embed_dimensions: usize,
    // It is only useful in training, since this project focuses on inference, this parameter do nothing.
    #[config(default = "1")]
    padding_idx: usize,
    #[config(default = "512")]
    max_position_embeddings: usize,
    #[config(default = "8")]
    decoder_attention_heads: usize,
    #[config(default = "384")]
    cross_attention_hidden_size: usize,
    #[config(default = "1024")]
    decoder_ffn_dim: usize,
    #[config(default = "0.0")]
    dropout: f64,
    #[config(default = "0.0")]
    attention_dropout: f64,
    #[config(default = "0.0")]
    activation_dropout: f64,
}

impl TrOCRForCausalLMConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TrOCRForCausalLM<B> {
        let model = TrOCRDecoderConfig::new()
            .with_decoder_layers(self.decoder_layers)
            .with_vocab_size(self.vocab_size)
            .with_embed_dimensions(self.embed_dimensions)
            .with_padding_idx(self.padding_idx)
            .with_max_position_embeddings(self.max_position_embeddings)
            .with_decoder_attention_heads(self.decoder_attention_heads)
            .with_cross_attention_hidden_size(self.cross_attention_hidden_size)
            .with_decoder_ffn_dim(self.decoder_ffn_dim)
            .with_dropout(self.dropout)
            .with_attention_dropout(self.attention_dropout)
            .with_activation_dropout(self.activation_dropout)
            .init(device);
        let output_projection = LinearConfig::new(self.embed_dimensions, self.vocab_size)
            .with_bias(false)
            .init(device);

        TrOCRForCausalLM {
            model,
            output_projection,
        }
    }
}

#[cfg(test)]
mod test {
    use burn::{
        backend::{libtorch::LibTorchDevice, LibTorch},
        module::{Module, Param},
        record::{FullPrecisionSettings, PrettyJsonFileRecorder},
        tensor::{Int, Tensor},
    };

    use crate::model::{
        deit::deit_model::{DeiTModel, DeiTModelConfig},
        trocr::decoder::TrOCRForCausalLMConfig,
    };

    use super::TrOCRDecoderLayerConfig;

    type Backend = LibTorch;
    const DEVICE: LibTorchDevice = LibTorchDevice::Cuda(0);

    fn init_tensor() -> Tensor<Backend, 4> {
        let tensor = Param::from_tensor(Tensor::<Backend, 4>::empty([1, 3, 384, 384], &DEVICE));
        let pfr = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();
        let tensor = tensor.load_file("./tensor.json", &pfr, &DEVICE).unwrap();
        let tensor = tensor.val();

        tensor
    }

    pub fn load_deit_model() -> DeiTModel<Backend> {
        let deit_model = DeiTModelConfig::new().init::<Backend>(&DEVICE);

        let pfr: PrettyJsonFileRecorder<FullPrecisionSettings> =
            PrettyJsonFileRecorder::<FullPrecisionSettings>::new();

        let deit_model = deit_model
            .load_file("./deit_model.json", &pfr, &DEVICE)
            .unwrap();

        deit_model
    }

    #[test]
    pub fn save() {
        let x = TrOCRDecoderLayerConfig::new().init::<Backend>(&DEVICE);

        let pfr: PrettyJsonFileRecorder<FullPrecisionSettings> =
            PrettyJsonFileRecorder::<FullPrecisionSettings>::new();

        x.save_file("./decoder_layer.json", &pfr).unwrap();
    }

    #[test]
    fn test_correctness() {
        let tensor = init_tensor();
        let deit_model = load_deit_model();
        let pfr = PrettyJsonFileRecorder::<FullPrecisionSettings>::new();
        let decoder = TrOCRForCausalLMConfig::new().init(&DEVICE);
        let decoder = decoder.load_file("./decoder.json", &pfr, &DEVICE).unwrap();
        // decoder.clone().save_file("./decoder.json", &pfr).unwrap();

        let encoder_res = deit_model.forward(tensor);

        let decoder_res: Tensor<Backend, 2, Int> = Tensor::zeros([1, 200], &DEVICE);
        let mut decoder_res = decoder_res.slice_assign(
            [0..1, 0..1],
            Tensor::<Backend, 2, Int>::from_ints([[1]], &DEVICE),
        );
        for i in 0..199 {
            let res = decoder
                .forward(decoder_res.clone(), encoder_res.clone(), vec![])
                .0;
            let idx = res.slice([0..1, i..(i + 1)]).argmax(2).reshape([1, 1]);
            decoder_res = decoder_res.slice_assign([0..1, (i + 1)..(i + 2)], idx);
        }

        println!("{}", decoder_res);
    }
}
