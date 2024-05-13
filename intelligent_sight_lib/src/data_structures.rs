use crate::unified_item::UnifiedItem;
use anyhow::Result;
use std::ops::{Deref, DerefMut};

#[derive(Clone)]
pub struct Image {
    pub width: u32,
    pub height: u32,
    data: UnifiedItem<u8>,
}

impl Image {
    pub fn new(width: u32, height: u32) -> Result<Self> {
        Ok(Image {
            width,
            height,
            data: UnifiedItem::new((width * height * 3) as usize)?, // 3 channels
        })
    }
}

impl Default for Image {
    fn default() -> Self {
        match Image::new(640, 640) {
            Ok(image) => image,
            Err(err) => {
                panic!(
                    "Failed to create default Image, allocation failure: {}",
                    err
                );
            }
        }
    }
}

impl Deref for Image {
    type Target = UnifiedItem<u8>;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for Image {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}
#[derive(Clone)]
pub struct Tensor {
    size: Vec<usize>,
    data: UnifiedItem<f32>,
}

impl Deref for Tensor {
    type Target = UnifiedItem<f32>;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl DerefMut for Tensor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl Tensor {
    pub fn new(size: Vec<usize>) -> Result<Self> {
        Ok(Tensor {
            data: UnifiedItem::new(size.iter().fold(1, |sum, num| sum * num))?,
            size,
        })
    }

    pub fn size(&self) -> &Vec<usize> {
        &self.size
    }
}
