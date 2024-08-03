use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        PaddingConfig2d,
    },
    tensor::{backend::Backend, Tensor},
};

pub struct EmbedFeatures<B: Backend> {
    pub features: Tensor<B, 3>,
    pub height: usize,
    pub width: usize,
}

impl<B: Backend> EmbedFeatures<B> {}

#[derive(Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    proj: Conv2d<B>,
    patch_size: [usize; 2],
    pub(crate) num_patches: usize,
}

impl<B: Backend> PatchEmbed<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> EmbedFeatures<B> {
        let x = self.proj.forward(input);
        let [_, _, height, width] = x.dims();
        // Shape: [batch_size, height * width, channels]
        let x = x.flatten(2, 3).swap_dims(1, 2);

        EmbedFeatures {
            features: x,
            height,
            width,
        }
    }
}

#[derive(Config)]
pub struct PatchEmbedConfig {
    #[config(default = "[384, 384]")]
    image_size: [usize; 2],
    #[config(default = "[16, 16]")]
    patch_size: [usize; 2],
    #[config(default = "3")]
    in_channels: usize,
    #[config(default = "384")]
    embed_dimensions: usize,
}

impl PatchEmbedConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> PatchEmbed<B> {
        let proj = Conv2dConfig::new([self.in_channels, self.embed_dimensions], self.patch_size)
            .with_stride(self.patch_size)
            .with_padding(PaddingConfig2d::Explicit(0, 0))
            .init(device);
        let num_patches =
            (self.image_size[0] / self.patch_size[0]) * (self.image_size[1] / self.patch_size[1]);

        PatchEmbed {
            proj,
            patch_size: self.patch_size,
            num_patches,
        }
    }
}
