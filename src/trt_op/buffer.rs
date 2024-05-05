use crate::trt_op::{cuda_free, cuda_malloc};
use std::ops::Deref;

pub struct Buffer {
    pub size: u32,
    pub data: FloatBuffer,
}

pub struct FloatBuffer(Vec<f32>);

impl FloatBuffer {
    fn new(size: usize) -> Self {
        FloatBuffer(cuda_malloc(size).expect("Failed to allocate memory"))
    }
}

impl Deref for FloatBuffer {
    type Target = Vec<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Drop for FloatBuffer {
    fn drop(&mut self) {
        cuda_free(&mut self.0).unwrap();
    }
}

impl Clone for FloatBuffer {
    fn clone(&self) -> Self {
        FloatBuffer(cuda_malloc(self.0.len()).expect("Failed to allocate memory"))
    }
}
