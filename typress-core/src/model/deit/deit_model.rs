use burn::{
    config::Config,
    module::Module,
    nn::{LayerNorm, LayerNormConfig},
    prelude::Backend,
    tensor::Tensor,
};

use super::{
    deit_embed::{DeiTEmbed, DeiTEmbedConfig},
    deit_encoder::{DeiTEncoder, DeiTEncoderConfig},
};

#[derive(Module, Debug)]
pub struct DeiTModel<B: Backend> {
    embed: DeiTEmbed<B>,
    encoder: DeiTEncoder<B>,
    layernorm: LayerNorm<B>,
}

impl<B: Backend> DeiTModel<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 3> {
        let x = self.embed.forward(input).features;
        let x = self.encoder.forward(x);
        let x = self.layernorm.forward(x);

        x
    }
}

#[derive(Config)]
pub struct DeiTModelConfig {
    #[config(default = "[384, 384]")]
    image_size: [usize; 2],
    #[config(default = "[16, 16]")]
    patch_size: [usize; 2],
    #[config(default = "3")]
    in_channels: usize,
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

impl DeiTModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DeiTModel<B> {
        let embed = DeiTEmbedConfig::new()
            .with_image_size(self.image_size)
            .with_patch_size(self.patch_size)
            .with_in_channels(self.in_channels)
            .with_embed_dimensions(self.embed_dimensions)
            .with_dropout_prob(self.dropout_prob)
            .init(device);
        let encoder = DeiTEncoderConfig::new()
            .with_embed_dimensions(self.embed_dimensions)
            .with_num_heads(self.num_heads)
            .with_num_layers(self.num_layers)
            .with_intermediate_dim(self.intermediate_dim)
            .with_dropout_prob(self.dropout_prob)
            .with_layer_norm_eps(self.layer_norm_eps)
            .init(device);
        let layernorm = LayerNormConfig::new(self.embed_dimensions)
            .with_epsilon(self.layer_norm_eps)
            .init(device);

        DeiTModel {
            embed,
            encoder,
            layernorm,
        }
    }
}

#[cfg(test)]
mod test {
    #[cfg(any(feature = "ndarray", feature = "tch"))]
    use super::{DeiTModel, DeiTModelConfig};
    #[cfg(any(feature = "ndarray", feature = "tch"))]
    use crate::image_processing::{ImageReader, NormalizeInfo, SizeInfo};
    #[cfg(any(feature = "ndarray", feature = "tch"))]
    use burn::{
        backend::{ndarray::NdArrayDevice, NdArray},
        module::Module,
        record::{BinFileRecorder, FullPrecisionSettings, PrettyJsonFileRecorder},
        tensor::Tensor,
    };

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

    #[test]
    #[cfg(any(feature = "ndarray", feature = "tch"))]
    pub fn save() {
        let x = DeiTModelConfig::new().init::<Backend>(&DEVICE);

        let pfr: PrettyJsonFileRecorder<FullPrecisionSettings> =
            PrettyJsonFileRecorder::<FullPrecisionSettings>::new();

        x.save_file("./deit_model.json", &pfr).unwrap();
    }

    #[test]
    #[cfg(any(feature = "ndarray", feature = "tch"))]
    fn test_correctness() {
        let tensor = init_tensor();
        let deit_model = load_deit_model();

        let tensor = deit_model.forward(tensor);
        println!("{}", tensor);
    }
}
