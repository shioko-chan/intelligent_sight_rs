pub struct Image {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl Image {
    pub fn new(width: u32, height: u32) -> Self {
        Image {
            width,
            height,
            data: vec![0; (width * height * 3) as usize], // 3 channels
        }
    }
}

impl Default for Image {
    fn default() -> Self {
        Image::new(0, 0)
    }
}

impl Clone for Image {
    fn clone(&self) -> Self {
        Image {
            width: self.width,
            height: self.height,
            data: self.data.clone(),
        }
    }
}
