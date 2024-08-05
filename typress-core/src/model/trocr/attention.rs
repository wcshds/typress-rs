use alloc::vec;
use burn::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    prelude::Backend,
    tensor::{activation::softmax, Tensor},
};

#[derive(Module, Debug)]
pub struct TrOCRAttention<B: Backend> {
    k_proj: Linear<B>,
    v_proj: Linear<B>,
    q_proj: Linear<B>,
    out_proj: Linear<B>,
    dropout: Dropout,
    num_heads: usize,
    head_dim: usize,
    scaling: f64,
}

impl<B: Backend> TrOCRAttention<B> {
    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        key_value_states: Option<Tensor<B, 3>>,
        attention_mask: Option<Tensor<B, 4>>,
        past_key_value: Option<(Tensor<B, 4>, Tensor<B, 4>)>,
    ) -> (Tensor<B, 3>, (Tensor<B, 4>, Tensor<B, 4>)) {
        let [batch_size, target_len, embed_dim] = hidden_states.dims();

        let query_states = self.q_proj.forward(hidden_states.clone()) * self.scaling;
        let [key_states, value_states] = if let Some(key_value_states) = key_value_states {
            // cross attention
            let (key_states, value_states);
            if let Some(past_key_value) = past_key_value {
                key_states = past_key_value.0;
                value_states = past_key_value.1;
            } else {
                key_states = self._shape(
                    self.k_proj.forward(key_value_states.clone()),
                    -1,
                    batch_size as i32,
                );
                value_states =
                    self._shape(self.v_proj.forward(key_value_states), -1, batch_size as i32);
            }

            [key_states, value_states]
        } else {
            // self attention
            let mut key_states = self._shape(
                self.k_proj.forward(hidden_states.clone()),
                -1,
                batch_size as i32,
            );
            let mut value_states =
                self._shape(self.v_proj.forward(hidden_states), -1, batch_size as i32);

            if let Some(past_key_value) = past_key_value {
                key_states = Tensor::cat(vec![past_key_value.0, key_states], 2);
                value_states = Tensor::cat(vec![past_key_value.1, value_states], 2);
            }

            [key_states, value_states]
        };

        let past_key_value = (key_states.clone(), value_states.clone());

        let proj_shape = [
            (batch_size * self.num_heads) as i32,
            -1,
            self.head_dim as i32,
        ];
        let query_states = query_states
            .reshape([batch_size, target_len, self.num_heads, self.head_dim])
            .swap_dims(1, 2)
            .reshape(proj_shape);
        let key_states = key_states.reshape(proj_shape);
        let value_states = value_states.reshape(proj_shape);

        let source_len = key_states.dims()[1];
        let mut attn_weights = query_states.matmul(key_states.swap_dims(1, 2));

        if let Some(attention_mask) = attention_mask {
            assert!(
                attention_mask.dims() == [batch_size, 1, target_len, source_len],
                "Attention mask should be of size {:?}, but is {:?}",
                [batch_size, 1, target_len, source_len],
                attention_mask.dims()
            );

            let tmp = attn_weights.reshape([batch_size, self.num_heads, target_len, source_len])
                + attention_mask;
            attn_weights = tmp.reshape([batch_size * self.num_heads, target_len, source_len]);
        }

        let attn_weights_last_dim = attn_weights.dims().len() - 1;
        let attn_weights = softmax(attn_weights, attn_weights_last_dim);

        let attn_probs = self.dropout.forward(attn_weights);

        let attn_output = attn_probs.matmul(value_states);

        let attn_output =
            attn_output.reshape([batch_size, self.num_heads, target_len, self.head_dim]);
        let attn_output = attn_output.swap_dims(1, 2);
        let attn_output = attn_output.reshape([batch_size, target_len, embed_dim]);

        let attn_output = self.out_proj.forward(attn_output);

        return (attn_output, past_key_value);
    }

    fn _shape(&self, tensor: Tensor<B, 3>, seq_length: i32, batch_size: i32) -> Tensor<B, 4> {
        tensor
            .reshape([
                batch_size,
                seq_length,
                self.num_heads as i32,
                self.head_dim as i32,
            ])
            .swap_dims(1, 2)
    }
}

#[derive(Config)]
pub struct TrOCRAttentionConfig {
    embed_dim: usize,
    num_heads: usize,
    #[config(default = "None")]
    kdim: Option<usize>,
    #[config(default = "None")]
    vdim: Option<usize>,
    #[config(default = "0.0")]
    dropout_prob: f64,
    #[config(default = "true")]
    bias: bool,
}

impl TrOCRAttentionConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TrOCRAttention<B> {
        let kdim = if let Some(content) = self.kdim {
            content
        } else {
            self.embed_dim
        };
        let vdim = if let Some(content) = self.vdim {
            content
        } else {
            self.embed_dim
        };
        let head_dim = self.embed_dim / self.num_heads;
        assert!(
            head_dim * self.num_heads == self.embed_dim,
            "embed_dim must be divisible by num_heads (got `embed_dim`: {} and `num_heads`: {}",
            self.embed_dim,
            self.num_heads
        );
        let scaling = (head_dim as f64).powf(-0.5);

        let dropout = DropoutConfig::new(self.dropout_prob).init();

        let k_proj = LinearConfig::new(kdim, self.embed_dim)
            .with_bias(self.bias)
            .init(device);
        let v_proj = LinearConfig::new(vdim, self.embed_dim)
            .with_bias(self.bias)
            .init(device);
        let q_proj = LinearConfig::new(self.embed_dim, self.embed_dim)
            .with_bias(self.bias)
            .init(device);

        let out_proj = LinearConfig::new(self.embed_dim, self.embed_dim)
            .with_bias(self.bias)
            .init(device);

        TrOCRAttention {
            k_proj,
            v_proj,
            q_proj,
            out_proj,
            dropout,
            num_heads: self.num_heads,
            head_dim,
            scaling,
        }
    }
}
