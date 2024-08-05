use alloc::vec::Vec;
use burn::{
    config::Config,
    module::Module,
    nn::{
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
        Dropout, DropoutConfig, Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig, Tanh,
    },
    prelude::Backend,
    tensor::Tensor,
};

#[derive(Module, Debug)]
pub struct DeiTIntermediate<B: Backend> {
    dense: Linear<B>,
    intermediate_act_fn: Gelu,
}

impl<B: Backend> DeiTIntermediate<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.dense.forward(input);
        let x = self.intermediate_act_fn.forward(x);

        x
    }
}

#[derive(Config)]
pub struct DeiTIntermediateConfig {
    // config.hidden_size
    #[config(default = "384")]
    embed_dimensions: usize,
    // config.intermediate_size
    #[config(default = "1536")]
    intermediate_dim: usize,
}

impl DeiTIntermediateConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DeiTIntermediate<B> {
        let dense =
            LinearConfig::new(self.embed_dimensions, self.intermediate_dim).init::<B>(device);
        let intermediate_act_fn = Gelu::new();

        DeiTIntermediate {
            dense,
            intermediate_act_fn,
        }
    }
}

#[derive(Module, Debug)]
pub struct DeiTOutput<B: Backend> {
    dense: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> DeiTOutput<B> {
    pub fn forward(&self, hidden: Tensor<B, 3>, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.dense.forward(hidden);
        let x = self.dropout.forward(x);
        // residual
        let x = x + input;

        x
    }
}

#[derive(Config)]
pub struct DeiTOutputConfig {
    // config.hidden_size
    #[config(default = "384")]
    embed_dimensions: usize,
    // config.intermediate_size
    #[config(default = "1536")]
    intermediate_dim: usize,
    #[config(default = "0.0")]
    dropout_prob: f64,
}

impl DeiTOutputConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DeiTOutput<B> {
        let dense =
            LinearConfig::new(self.intermediate_dim, self.embed_dimensions).init::<B>(device);
        let dropout = DropoutConfig::new(self.dropout_prob).init();

        DeiTOutput { dense, dropout }
    }
}

#[derive(Module, Debug)]
pub struct DeiTLayer<B: Backend> {
    attention: MultiHeadAttention<B>,
    intermediate: DeiTIntermediate<B>,
    output: DeiTOutput<B>,
    layernorm_before: LayerNorm<B>,
    layernorm_after: LayerNorm<B>,
}

impl<B: Backend> DeiTLayer<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = input.clone();
        let x = self.layernorm_before.forward(input);
        let x = self.attention.forward(MhaInput::self_attn(x)).context;
        // first residual connection
        let x = x + residual.clone();
        let residual = x.clone();
        // in DeiT, layernorm is also applied after self-attention
        let x = self.layernorm_after.forward(x);
        let x = self.intermediate.forward(x);
        // second residual connection is done here
        let x = self.output.forward(x, residual);

        x
    }
}

#[derive(Config)]
pub struct DeiTLayerConfig {
    // config.hidden_size
    #[config(default = "384")]
    embed_dimensions: usize,
    // config.n_heads
    #[config(default = "6")]
    num_heads: usize,
    // config.intermediate_size
    #[config(default = "1536")]
    intermediate_dim: usize,
    #[config(default = "0.0")]
    dropout_prob: f64,
    #[config(default = "1e-12")]
    layer_norm_eps: f64,
}

impl DeiTLayerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DeiTLayer<B> {
        let attention = MultiHeadAttentionConfig::new(self.embed_dimensions, self.num_heads)
            .with_dropout(self.dropout_prob)
            .init(device);
        let intermediate = DeiTIntermediateConfig::new()
            .with_embed_dimensions(self.embed_dimensions)
            .with_intermediate_dim(self.intermediate_dim)
            .init(device);
        let output = DeiTOutputConfig::new()
            .with_embed_dimensions(self.embed_dimensions)
            .with_intermediate_dim(self.intermediate_dim)
            .with_dropout_prob(self.dropout_prob)
            .init(device);
        let layernorm_before = LayerNormConfig::new(self.embed_dimensions)
            .with_epsilon(self.layer_norm_eps)
            .init(device);
        let layernorm_after = LayerNormConfig::new(self.embed_dimensions)
            .with_epsilon(self.layer_norm_eps)
            .init(device);

        DeiTLayer {
            attention,
            intermediate,
            output,
            layernorm_before,
            layernorm_after,
        }
    }
}

#[derive(Module, Debug)]
pub struct DeiTEncoder<B: Backend> {
    layers: Vec<DeiTLayer<B>>,
}

impl<B: Backend> DeiTEncoder<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let mut x = input;
        for layer in &self.layers {
            x = layer.forward(x);
        }

        x
    }
}

#[derive(Config)]
pub struct DeiTEncoderConfig {
    // config.hidden_size
    #[config(default = "384")]
    embed_dimensions: usize,
    // config.n_heads
    #[config(default = "6")]
    num_heads: usize,
    // config.num_hidden_layers
    #[config(default = "12")]
    num_layers: usize,
    // config.intermediate_size
    #[config(default = "1536")]
    intermediate_dim: usize,
    #[config(default = "0.0")]
    dropout_prob: f64,
    #[config(default = "1e-12")]
    layer_norm_eps: f64,
}

impl DeiTEncoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DeiTEncoder<B> {
        let layers = (0..self.num_layers)
            .map(|_| {
                DeiTLayerConfig::new()
                    .with_embed_dimensions(self.embed_dimensions)
                    .with_num_heads(self.num_heads)
                    .with_intermediate_dim(self.intermediate_dim)
                    .with_dropout_prob(self.dropout_prob)
                    .with_layer_norm_eps(self.layer_norm_eps)
                    .init(device)
            })
            .collect();

        DeiTEncoder { layers }
    }
}

#[derive(Module, Debug)]
pub struct DeiTPooler<B: Backend> {
    dense: Linear<B>,
    activation: Tanh,
}

impl<B: Backend> DeiTPooler<B> {
    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        // We "pool" the model by simply taking the hidden state corresponding
        // to the first token.
        let first_token_tensor = input.slice([None, Some((0, 1))]);
        let pooled_output = self.dense.forward(first_token_tensor);
        let pooled_output = self.activation.forward(pooled_output);

        pooled_output
    }
}

#[derive(Config)]
pub struct DeiTPoolerConfig {
    // config.hidden_size
    #[config(default = "384")]
    embed_dimensions: usize,
}

impl DeiTPoolerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DeiTPooler<B> {
        let dense = LinearConfig::new(self.embed_dimensions, self.embed_dimensions).init(device);

        DeiTPooler {
            dense,
            activation: Tanh::new(),
        }
    }
}
