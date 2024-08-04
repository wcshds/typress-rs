use burn::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, Relu},
    prelude::Backend,
    tensor::Tensor,
};

use super::attention::{TrOCRAttention, TrOCRAttentionConfig};

#[derive(Module, Debug)]
pub struct TrOCRDecoderLayer<B: Backend> {
    self_attn: TrOCRAttention<B>,
    activation_fn: Relu,
    self_attn_layer_norm: LayerNorm<B>,
    encoder_attn: TrOCRAttention<B>,
    encoder_attn_layer_norm: LayerNorm<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    final_layer_norm: LayerNorm<B>,
    dropout: Dropout,
    activation_dropout: Dropout,
}

impl<B: Backend> TrOCRDecoderLayer<B> {
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 4>>,
        encoder_hidden_states: Option<Tensor<B, 3>>,
        encoder_attention_mask: Option<Tensor<B, 4>>,
        past_key_value: (
            Option<(Tensor<B, 4>, Tensor<B, 4>)>,
            Option<(Tensor<B, 4>, Tensor<B, 4>)>,
        ),
    ) -> (
        // hidden_states
        Tensor<B, 3>,
        // present_key_value
        (
            Option<(Tensor<B, 4>, Tensor<B, 4>)>,
            Option<(Tensor<B, 4>, Tensor<B, 4>)>,
        ),
    ) {
        let residual = hidden_states.clone();

        // decoder uni-directional self-attention cached key/values tuple
        let self_attn_past_key_value = past_key_value.0;
        // add present self-attn cache to first present_key_value tuple
        let (hidden_states, present_key_value) = self.self_attn.forward(
            hidden_states,
            None,
            attention_mask,
            self_attn_past_key_value,
        );
        let hidden_states = self.dropout.forward(hidden_states);
        let hidden_states = residual + hidden_states;
        let mut hidden_states = self.self_attn_layer_norm.forward(hidden_states);

        let mut present_key_value_full = (Some(present_key_value), None);
        if let Some(encoder_hidden_states) = encoder_hidden_states {
            let residual = hidden_states.clone();

            // cross_attn cached key/values tuple
            let cross_attn_past_key_value = past_key_value.1;
            let (tmp, cross_attn_present_key_value) = self.encoder_attn.forward(
                hidden_states,
                Some(encoder_hidden_states),
                encoder_attention_mask,
                cross_attn_past_key_value,
            );
            hidden_states = self.dropout.forward(tmp);
            hidden_states = residual + hidden_states;
            hidden_states = self.encoder_attn_layer_norm.forward(hidden_states);

            // add cross-attn to present_key_value_full tuple
            present_key_value_full.1 = Some(cross_attn_present_key_value)
        }

        // Fully Connected
        let residual = hidden_states.clone();
        let hidden_states = self.fc1.forward(hidden_states);
        let hidden_states = self.activation_fn.forward(hidden_states);
        let hidden_states = self.activation_dropout.forward(hidden_states);
        let hidden_states = self.fc2.forward(hidden_states);

        let hidden_states = self.dropout.forward(hidden_states);
        let hidden_states = residual + hidden_states;
        let hidden_states = self.final_layer_norm.forward(hidden_states);

        (hidden_states, present_key_value_full)
    }
}

#[derive(Config)]
pub struct TrOCRDecoderLayerConfig {
    #[config(default = "256")]
    embed_dimensions: usize,
    // config.decoder_attention_heads
    #[config(default = "8")]
    decoder_attention_heads: usize,
    #[config(default = "384")]
    cross_attention_hidden_size: usize,
    #[config(default = "1024")]
    decoder_ffn_dim: usize,
    #[config(default = "0.0")]
    attention_dropout: f64,
    #[config(default = "0.0")]
    activation_dropout: f64,
}

impl TrOCRDecoderLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TrOCRDecoderLayer<B> {
        let self_attn =
            TrOCRAttentionConfig::new(self.embed_dimensions, self.decoder_attention_heads)
                .with_kdim(None)
                .with_vdim(None)
                .with_bias(true)
                .with_dropout_prob(self.attention_dropout)
                .init(device);
        let dropout = DropoutConfig::new(self.attention_dropout).init();
        let activation_dropout = DropoutConfig::new(self.activation_dropout).init();
        let activation_fn = Relu::new();
        let self_attn_layer_norm = LayerNormConfig::new(self.embed_dimensions).init(device);
        let encoder_attn =
            TrOCRAttentionConfig::new(self.embed_dimensions, self.decoder_attention_heads)
                .with_kdim(Some(self.cross_attention_hidden_size))
                .with_vdim(Some(self.cross_attention_hidden_size))
                .with_bias(true)
                .with_dropout_prob(self.attention_dropout)
                .init(device);
        let encoder_attn_layer_norm = LayerNormConfig::new(self.embed_dimensions).init(device);
        let fc1 = LinearConfig::new(self.embed_dimensions, self.decoder_ffn_dim).init(device);
        let fc2 = LinearConfig::new(self.decoder_ffn_dim, self.embed_dimensions).init(device);
        let final_layer_norm = LayerNormConfig::new(self.embed_dimensions).init(device);

        TrOCRDecoderLayer {
            self_attn,
            activation_fn,
            self_attn_layer_norm,
            encoder_attn,
            encoder_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
            dropout,
            activation_dropout,
        }
    }
}
