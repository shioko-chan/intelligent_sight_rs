mod buffer;

use anyhow::{anyhow, Result};
use std::mem;

pub use buffer::Buffer;
mod trt_op_ffi {
    use std::ffi::c_char;
    extern "C" {
        pub fn cuda_malloc(size: u32, ptr: *mut *mut u8) -> u8;
        pub fn cuda_free(ptr: *mut u8) -> u8;
        pub fn create_engine(engine_filename: *const c_char, width: u32, height: u32) -> u8;
    }
}

pub fn cuda_malloc(size: usize) -> Result<Vec<f32>> {
    let mut ptr = std::ptr::null_mut();
    match unsafe {
        trt_op_ffi::cuda_malloc(
            (size * mem::size_of::<f32>() / mem::size_of::<u8>()) as u32,
            &mut ptr as *mut *mut f32 as *mut *mut u8,
        )
    } {
        0 => unsafe { Ok(Vec::from_raw_parts(ptr, size, size)) },
        err => Err(anyhow!("Failed to allocate memory, code: {}", err)),
    }
}

pub fn cuda_free(vec: &mut Vec<f32>) -> Result<()> {
    let ptr = vec.as_mut_ptr();
    match unsafe { trt_op_ffi::cuda_free(ptr as *mut u8) } {
        0 => Ok(()),
        err => Err(anyhow!("Failed to free memory, code: {}", err)),
    }
}

pub fn create_engine(engine_filename: &'static str, width: u32, height: u32) -> Result<()> {
    let engine_filename = std::ffi::CString::new(engine_filename)?;
    match unsafe { trt_op_ffi::create_engine(engine_filename.as_ptr(), width, height) } {
        0 => Ok(()),
        err => Err(anyhow!("Failed to create engine, code: {}", err)),
    }
}
