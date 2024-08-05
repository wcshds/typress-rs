use std::path::Path;

use alloc::vec::Vec;

use burn::tensor::{backend::Backend, Shape, Tensor, TensorData};

#[derive(Debug)]
pub struct ImageReader {
    imgs_raw_data: Vec<u8>,
    sizes: Vec<SizeInfo>,
    batch: usize,
    has_same_size: bool,
}

#[derive(Debug, Clone, Copy)]
pub struct SizeInfo {
    height: usize,
    width: usize,
}

impl SizeInfo {
    pub fn new(height: usize, width: usize) -> Self {
        Self { height, width }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct NormalizeInfo {
    mean: [f32; 3],
    std: [f32; 3],
}

impl NormalizeInfo {
    pub fn new(mean: [f32; 3], std: [f32; 3]) -> Self {
        Self { mean, std }
    }
}

impl ImageReader {
    /// Read images from the paths and resize these images.
    ///
    /// ## Parameters:
    ///
    /// - resize: Size of the image after resizing.
    pub fn read_images<P: AsRef<Path>>(paths: &[P], resize: Option<SizeInfo>) -> ImageReader {
        let batch = paths.len();
        assert!(batch >= 1, "`paths` should contain at least one path.");

        let mut total_img_vec = if let Some(SizeInfo { height, width }) = resize {
            Vec::with_capacity(batch * height * width * 3)
        } else {
            Vec::new()
        };
        let mut sizes = Vec::with_capacity(batch);
        let mut has_same_size = true;

        for path in paths {
            let img = image::open(path)
                .expect(&format!("Cannot read image from path: {:?}", path.as_ref()));
            let (resized_img, size) = if let Some(SizeInfo { height, width }) = resize {
                let img = img.resize_exact(
                    width as u32,
                    height as u32,
                    image::imageops::FilterType::CatmullRom,
                );
                let size = SizeInfo { width, height };
                (img, size)
            } else {
                let size = SizeInfo {
                    width: img.width() as usize,
                    height: img.height() as usize,
                };
                (img, size)
            };

            let mut img_vec = resized_img.into_rgb8().into_vec();
            total_img_vec.append(&mut img_vec);
            sizes.push(size.clone());

            if size.height != sizes[0].height || size.width != sizes[0].width {
                has_same_size = false;
            }
        }

        Self {
            imgs_raw_data: total_img_vec,
            sizes,
            batch,
            has_same_size,
        }
    }

    /// Convert images to a tensor.
    ///
    /// ## Parameters:
    ///
    /// - rescale: Scale factor to rescale the image.
    ///
    /// - normalize: Mean and std to Mean to normalize the image.
    pub fn to_tensor<B: Backend>(
        self,
        device: &B::Device,
        rescale: Option<f32>,
        normalize: Option<NormalizeInfo>,
    ) -> Tensor<B, 4> {
        assert!(self.has_same_size, "All images should have the same size.");

        let data = TensorData::new(
            self.imgs_raw_data,
            Shape::new([self.batch, self.sizes[0].height, self.sizes[0].width, 3]),
        );
        let tensor: Tensor<B, 4> = Tensor::from_data(data, device);
        let mut tensor = tensor.permute([0, 3, 1, 2]);
        if let Some(scale) = rescale {
            tensor = tensor * scale;
        }
        if let Some(NormalizeInfo { mean, std }) = normalize {
            tensor = (tensor - Tensor::<B, 1>::from_floats(mean, device).reshape([1, 3, 1, 1]))
                / Tensor::<B, 1>::from_floats(std, device).reshape([1, 3, 1, 1]);
        }

        tensor
    }
}
