mod err_code;
use crate::cuda_op::err_code::CUDA_ERR_NAME;
use crate::UnifiedItem;
use anyhow::{anyhow, Result};
use err_code::TRT_ERR_NAME;

mod trt_op_ffi {
    use std::ffi::c_char;
    extern "C" {
        pub fn create_engine(engine_filename: *const c_char, width: u32, height: u32) -> u16;
        pub fn create_context() -> u16;
        pub fn infer(input_buffer: *const f32, output_buffer: *mut f32) -> u16;
        pub fn release_resources() -> u16;
    }
}

pub fn create_engine(engine_filename: &str, width: u32, height: u32) -> Result<()> {
    let engine_filename = std::ffi::CString::new(engine_filename)?;
    match unsafe { trt_op_ffi::create_engine(engine_filename.as_ptr(), width, height) } {
        0 => Ok(()),
        err @ 1..=9999 => Err(anyhow!(
            "Failed to create engine, code: {} ({})",
            err,
            CUDA_ERR_NAME
                .get(err as usize)
                .unwrap_or(&"err code unknown")
        )),
        err => Err(anyhow!(
            "Failed to create engine, code: {} ({})",
            err,
            TRT_ERR_NAME
                .get(err as usize)
                .unwrap_or(&"err code unknown")
        )),
    }
}

pub fn create_context() -> Result<()> {
    match unsafe { trt_op_ffi::create_context() } {
        0 => Ok(()),
        err @ 1..=9999 => Err(anyhow!(
            "Failed to create engine, code: {} ({})",
            err,
            CUDA_ERR_NAME
                .get(err as usize)
                .unwrap_or(&"err code unknown")
        )),
        err => Err(anyhow!(
            "Failed to create engine, code: {} ({})",
            err,
            TRT_ERR_NAME
                .get(err as usize)
                .unwrap_or(&"err code unknown")
        )),
    }
}

pub fn infer(input_buffer: &UnifiedItem<f32>, output_buffer: &mut UnifiedItem<f32>) -> Result<()> {
    match unsafe { trt_op_ffi::infer(input_buffer.as_ptr(), output_buffer.as_mut_ptr()) } {
        0 => Ok(()),
        err @ 1..=9999 => Err(anyhow!(
            "Failed to infer, code: {} ({})",
            err,
            CUDA_ERR_NAME
                .get(err as usize)
                .unwrap_or(&"err code unknown")
        )),
        err => Err(anyhow!(
            "Failed to infer, code: {} ({})",
            err,
            TRT_ERR_NAME
                .get(err as usize)
                .unwrap_or(&"err code unknown")
        )),
    }
}

pub fn release_resources() -> Result<()> {
    match unsafe { trt_op_ffi::release_resources() } {
        0 => Ok(()),
        err @ 1..=9999 => Err(anyhow!(
            "Failed to destroy engine, code: {} ({})",
            err,
            CUDA_ERR_NAME
                .get(err as usize)
                .unwrap_or(&"err code unknown")
        )),
        err => Err(anyhow!(
            "Failed to destroy engine, code: {} ({})",
            err,
            TRT_ERR_NAME
                .get(err as usize)
                .unwrap_or(&"err code unknown")
        )),
    }
}
