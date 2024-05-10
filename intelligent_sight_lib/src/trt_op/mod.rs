mod err_code;
use crate::unified_item::CArray;
use anyhow::{anyhow, Result};
use err_code::TRT_ERR_NAME;

mod trt_op_ffi {
    use std::ffi::c_char;
    extern "C" {
        pub fn create_engine(engine_filename: *const c_char, width: u32, height: u32) -> u8;
        pub fn infer(input_buffer: *const f32, output_buffer: *mut f32) -> u8;
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
