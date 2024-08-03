use burn::{
    config::Config,
    module::Module,
    nn::{Embedding, EmbeddingConfig},
    prelude::Backend,
    tensor::{Int, Tensor},
};

#[derive(Module, Debug)]
pub struct TrOCRScaledWordEmbedding<B: Backend> {
    embed: Embedding<B>,
    embed_scale: f64,
}

impl<B: Backend> TrOCRScaledWordEmbedding<B> {
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let x = self.embed.forward(input_ids);
        let x = x * self.embed_scale;

        x
    }
}

#[derive(Config)]
pub struct TrOCRScaledWordEmbeddingConfig {
    #[config(default = "1200")]
    num_embed: usize,
    #[config(default = "256")]
    embed_dimensions: usize,
    // It is only useful in training, since this project focuses on inference, this parameter do nothing.
    #[config(default = "1")]
    padding_idx: usize,
    #[config(default = "16.0")]
    embed_scale: f64,
}

impl TrOCRScaledWordEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TrOCRScaledWordEmbedding<B> {
        let embed = EmbeddingConfig::new(self.num_embed, self.embed_dimensions).init(device);

        TrOCRScaledWordEmbedding {
            embed,
            embed_scale: self.embed_scale,
        }
    }
}

#[derive(Module, Debug)]
pub struct TrOCRLearnedPositionalEmbedding<B: Backend> {
    embed: Embedding<B>,
    offset: usize,
}

impl<B: Backend> TrOCRLearnedPositionalEmbedding<B> {
    pub fn forward(&self, input: Tensor<B, 3>, past_key_values_length: usize) -> Tensor<B, 3> {
        let device = input.device();

        let [batch_size, seq_length, _] = input.dims();
        // let positions: Tensor<B, 2, Int> = Tensor::arange(
        //     (past_key_values_length as i64)..((past_key_values_length + seq_length) as i64),
        //     &device,
        // )
        // .expand([batch_size as i32, -1]);
        let positions: Tensor<B, 2, Int> = Tensor::arange(
            (past_key_values_length as i64)..((past_key_values_length + seq_length) as i64),
            &device,
        )
        .reshape([1, seq_length])
        .repeat(0, batch_size);

        let x = self.embed.forward(positions + self.offset as i32);
        x
    }
}

#[derive(Config)]
pub struct TrOCRLearnedPositionalEmbeddingConfig {
    #[config(default = "512")]
    num_embed: usize,
    #[config(default = "256")]
    embed_dimensions: usize,
    #[config(default = "2")]
    offset: usize,
}

impl TrOCRLearnedPositionalEmbeddingConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TrOCRLearnedPositionalEmbedding<B> {
        let embed =
            EmbeddingConfig::new(self.num_embed + self.offset, self.embed_dimensions).init(device);

        TrOCRLearnedPositionalEmbedding {
            embed,
            offset: self.offset,
        }
    }
}
