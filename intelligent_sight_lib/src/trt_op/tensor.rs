use crate::unified_item::UnifiedItem;
use anyhow::Result;

#[derive(Clone)]
pub struct Tensor {
    pub size: u32,
    pub data: UnifiedItem<f32>,
}

impl Tensor {
    pub fn new(size: u32) -> Result<Self> {
        Ok(Tensor {
            size,
            data: UnifiedItem::new(size as usize)?,
        })
    }
}
