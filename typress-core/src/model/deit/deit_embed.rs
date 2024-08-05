use alloc::vec;
use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Dropout, DropoutConfig},
    tensor::{backend::Backend, Tensor},
};

use super::patch_embed::{EmbedFeatures, PatchEmbed, PatchEmbedConfig};

#[derive(Module, Debug)]
pub struct DeiTEmbed<B: Backend> {
    patch_embed: PatchEmbed<B>,
    cls_token: Param<Tensor<B, 3>>,
    distillation_token: Param<Tensor<B, 3>>,
    position_embed: Param<Tensor<B, 3>>,
    dropout: Dropout,
}

impl<B: Backend> DeiTEmbed<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> EmbedFeatures<B> {
        let features = self.patch_embed.forward(input);
        let [height, width] = [features.height, features.width];
        let x = features.features;
        let [batch_size, _, _] = x.dims();

        let cls_tokens = self.cls_token.val().expand([batch_size as i32, -1, -1]);
        let distillation_tokens = self
            .distillation_token
            .val()
            .expand([batch_size as i32, -1, -1]);

        let x = Tensor::cat(vec![cls_tokens, distillation_tokens, x], 1);
        let position_embed = self.position_embed.val();

        let x = x + position_embed;

        let x = self.dropout.forward(x);

        EmbedFeatures {
            features: x,
            height,
            width,
        }
    }
}

#[derive(Config)]
pub struct DeiTEmbedConfig {
    #[config(default = "[384, 384]")]
    image_size: [usize; 2],
    #[config(default = "[16, 16]")]
    patch_size: [usize; 2],
    #[config(default = "3")]
    in_channels: usize,
    #[config(default = "384")]
    embed_dimensions: usize,
    #[config(default = "0.0")]
    dropout_prob: f64,
}

impl DeiTEmbedConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DeiTEmbed<B> {
        let cls_token =
            Param::from_tensor(Tensor::<B, 3>::zeros([1, 1, self.embed_dimensions], device));
        let distillation_token =
            Param::from_tensor(Tensor::<B, 3>::zeros([1, 1, self.embed_dimensions], device));
        let patch_embed = PatchEmbedConfig::new()
            .with_image_size(self.image_size)
            .with_in_channels(self.in_channels)
            .with_patch_size(self.patch_size)
            .with_embed_dimensions(self.embed_dimensions)
            .init::<B>(device);
        let num_patches = patch_embed.num_patches;
        let position_embed = Param::from_tensor(Tensor::<B, 3>::zeros(
            [1, num_patches + 2, self.embed_dimensions],
            device,
        ));
        let dropout = DropoutConfig::new(self.dropout_prob).init();

        DeiTEmbed {
            cls_token,
            distillation_token,
            patch_embed,
            position_embed,
            dropout,
        }
    }
}
