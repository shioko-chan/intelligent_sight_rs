mod err_code;
mod tensor;
use crate::unified_item::CArray;
use anyhow::{anyhow, Result};
use err_code::{CUDA_ERR_NAME, TRT_ERR_NAME};
use std::mem;

pub use tensor::Tensor;

mod trt_op_ffi {
    use std::ffi::c_char;
    extern "C" {
        pub fn cuda_malloc(size: u32, ptr: *mut *mut u8) -> u16;
        pub fn cuda_free(ptr: *mut u8) -> u16;
        pub fn create_engine(engine_filename: *const c_char, width: u32, height: u32) -> u8;
        pub fn infer(input_buffer: *const f32, output_buffer: *mut f32) -> u8;
    }
}

pub fn cuda_malloc<T>(size: usize) -> Result<CArray<T>>
where
    T: Sized,
{
    let mut ptr = std::ptr::null_mut();
    match unsafe {
        trt_op_ffi::cuda_malloc(
            (size * mem::size_of::<T>() / mem::size_of::<u8>()) as u32,
            &mut ptr as *mut *mut T as *mut *mut u8,
        )
    } {
        0 => Ok(CArray::from_raw_parts(ptr, size)),
        err => Err(anyhow!(
            "Failed to allocate memory, code: {} ({})",
            err,
            CUDA_ERR_NAME
                .get(err as usize)
                .unwrap_or(&"err code unknown")
        )),
    }
}

pub fn cuda_free<T>(vec: &mut CArray<T>) -> Result<()> {
    let ptr = unsafe { vec.as_mut_ptr() };
    match unsafe { trt_op_ffi::cuda_free(ptr as *mut u8) } {
        0 => Ok(()),
        err => Err(anyhow!(
            "Failed to free memory, code: {} ({})",
            err,
            CUDA_ERR_NAME
                .get(err as usize)
                .unwrap_or(&"err code unknown")
        )),
    }
}

pub fn create_engine(engine_filename: &'static str, width: u32, height: u32) -> Result<()> {
    let engine_filename = std::ffi::CString::new(engine_filename)?;
    match unsafe { trt_op_ffi::create_engine(engine_filename.as_ptr(), width, height) } {
        0 => Ok(()),
        err => Err(anyhow!(
            "Failed to create engine, code: {} ({})",
            err,
            TRT_ERR_NAME
                .get(err as usize)
                .unwrap_or(&"err code unknown")
        )),
    }
}

pub fn infer(input_buffer: &CArray<f32>, output_buffer: &mut CArray<f32>) -> Result<()> {
    match unsafe { trt_op_ffi::infer(input_buffer.as_ptr(), output_buffer.as_mut_ptr()) } {
        0 => Ok(()),
        err => Err(anyhow!(
            "Failed to infer, code: {} ({})",
            err,
            TRT_ERR_NAME
                .get(err as usize)
                .unwrap_or(&"err code unknown")
        )),
    }
}
