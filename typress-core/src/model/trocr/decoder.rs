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
        attention_mask: Option<Tensor<B, 4>>,
    ) -> (
        // hidden_states
        Tensor<B, 3>,
        // next_decoder_cache
        Vec<(
            Option<(Tensor<B, 4>, Tensor<B, 4>)>,
            Option<(Tensor<B, 4>, Tensor<B, 4>)>,
        )>,
    ) {
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
                attention_mask.clone(),
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
        attention_mask: Option<Tensor<B, 4>>,
    ) -> (
        // logits
        Tensor<B, 3>,
        // past_key_values
        Vec<(
            Option<(Tensor<B, 4>, Tensor<B, 4>)>,
            Option<(Tensor<B, 4>, Tensor<B, 4>)>,
        )>,
    ) {
        let outputs = self.model.forward(
            input_ids,
            encoder_hidden_states,
            past_key_values,
            attention_mask,
        );
        let logits = self.output_projection.forward(outputs.0);

        (logits, outputs.1)
    }

    fn generate_with_cache(
        &self,
        encoder_res: Tensor<B, 3>,
        max_iteration: usize,
    ) -> Tensor<B, 2, Int> {
        let batch_size = encoder_res.dims()[0];
        let device = encoder_res.device();

        let mut idx: Tensor<B, 2, Int> = Tensor::ones([batch_size, 1], &device);
        let mut past_key_values = Vec::with_capacity(0);
        let mut res_ids = Tensor::ones([batch_size, 1], &device);
        for i in 0..max_iteration {
            let (res, next_cache) =
                self.forward(idx.clone(), encoder_res.clone(), past_key_values, None);
            idx = res.argmax(2).flatten(1, 2);
            // Python Code:
            //   past_key_values = [
            //       (each[0][:, :, 0 : i + 1, :], each[1][:, :, 0 : i + 1, :], each[2], each[3])
            //       for each in past_key_values
            //   ]
            past_key_values = next_cache
                .into_iter()
                .map(|(self_attn_reserve, cross_attn_reserve)| {
                    let self_attn_reserve = self_attn_reserve.unwrap();
                    let self_attn_key =
                        self_attn_reserve
                            .0
                            .slice([None, None, Some((0, (i + 1) as i64)), None]);
                    let self_attn_value =
                        self_attn_reserve
                            .1
                            .slice([None, None, Some((0, (i + 1) as i64)), None]);
                    (Some((self_attn_key, self_attn_value)), cross_attn_reserve)
                })
                .collect();
            res_ids = Tensor::cat(vec![res_ids, idx.clone()], 1);

            if Tensor::all(idx.clone().equal_elem(2))
                .to_data()
                .as_slice::<bool>()
                .unwrap()
                == &[true]
            {
                break;
            };
        }

        res_ids
    }

    pub fn generate_without_cache(
        &self,
        encoder_res: Tensor<B, 3>,
        max_iteration: usize,
    ) -> Tensor<B, 2, Int> {
        let batch_size = encoder_res.dims()[0];
        let device = encoder_res.device();
        let target_len = max_iteration + 1;

        let res_ids: Tensor<B, 2, Int> = Tensor::zeros([batch_size, target_len], &device);
        let mut res_ids = res_ids.slice_assign(
            [0..batch_size, 0..1],
            Tensor::<B, 2, Int>::from_ints([[1]], &device).expand([batch_size as i32, -1]),
        );
        let attention_mask = Tensor::triu(
            Tensor::<B, 2>::ones([target_len, target_len], &device) * (-3.4028234663852886e38),
            1,
        )
        .reshape([1, 1, target_len, target_len])
        .expand([batch_size as i32, -1, -1, -1]);

        for i in 0..max_iteration {
            let (res, _) = self.forward(
                res_ids.clone(),
                encoder_res.clone(),
                Vec::with_capacity(0),
                Some(attention_mask.clone()),
            );
            let idx = res
                .slice([0..batch_size, i..(i + 1)])
                .argmax(2)
                .reshape([batch_size, 1]);
            res_ids = res_ids.slice_assign([0..batch_size, (i + 1)..(i + 2)], idx.clone());

            if Tensor::all(idx.equal_elem(2))
                .to_data()
                .as_slice::<bool>()
                .unwrap()
                == &[true]
            {
                break;
            };
        }

        res_ids
    }

    pub fn generate(
        &self,
        encoder_res: Tensor<B, 3>,
        max_iteration: usize,
        use_cache: bool,
    ) -> Tensor<B, 2, Int> {
        if use_cache {
            self.generate_with_cache(encoder_res, max_iteration)
        } else {
            self.generate_without_cache(encoder_res, max_iteration)
        }
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
    #[cfg(any(feature = "ndarray", feature = "tch"))]
    use super::TrOCRForCausalLM;
    #[cfg(any(feature = "ndarray", feature = "tch"))]
    use crate::{
        image_processing::{ImageReader, NormalizeInfo, SizeInfo},
        model::{
            deit::deit_model::{DeiTModel, DeiTModelConfig},
            trocr::decoder::TrOCRForCausalLMConfig,
        },
    };
    #[cfg(any(feature = "ndarray", feature = "tch"))]
    use burn::{
        module::Module,
        record::{BinFileRecorder, FullPrecisionSettings},
        tensor::Tensor,
    };

    #[cfg(feature = "tch")]
    use burn::backend::{libtorch::LibTorchDevice, LibTorch};
    #[cfg(feature = "tch")]
    type Backend = LibTorch;
    #[cfg(feature = "tch")]
    const DEVICE: LibTorchDevice = LibTorchDevice::Cuda(0);

    #[cfg(feature = "ndarray")]
    use burn::backend::{ndarray::NdArrayDevice, NdArray};
    #[cfg(feature = "ndarray")]
    type Backend = NdArray;
    #[cfg(feature = "ndarray")]
    const DEVICE: NdArrayDevice = NdArrayDevice::Cpu;

    #[cfg(any(feature = "ndarray", feature = "tch"))]
    fn init_tensor() -> Tensor<Backend, 4> {
        let image = ImageReader::read_images(&["../images/01.png"], Some(SizeInfo::new(384, 384)));
        let tensor = image.to_tensor(
            &DEVICE,
            Some(1.0 / 255.0),
            Some(NormalizeInfo::new([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])),
        );

        tensor
    }

    #[cfg(any(feature = "ndarray", feature = "tch"))]
    fn load_deit_model() -> DeiTModel<Backend> {
        let bfr = BinFileRecorder::<FullPrecisionSettings>::new();
        let deit_model = DeiTModelConfig::new().init::<Backend>(&DEVICE);
        let deit_model = deit_model
            .load_file("../weights/deit_model.bin", &bfr, &DEVICE)
            .unwrap();

        deit_model
    }

    #[cfg(any(feature = "ndarray", feature = "tch"))]
    fn load_decoder() -> TrOCRForCausalLM<Backend> {
        let bfr = BinFileRecorder::<FullPrecisionSettings>::new();
        let decoder = TrOCRForCausalLMConfig::new().init(&DEVICE);
        let decoder = decoder
            .load_file("../weights/decoder.bin", &bfr, &DEVICE)
            .unwrap();

        decoder
    }

    #[test]
    #[cfg(any(feature = "ndarray", feature = "tch"))]
    fn test_correctness() {
        let tensor = init_tensor();
        let deit_model = load_deit_model();
        let decoder = load_decoder();

        let encoder_res = deit_model.forward(tensor);
        let res_ids = decoder.generate(encoder_res, 200, false);

        println!("{}", res_ids);
    }
}
