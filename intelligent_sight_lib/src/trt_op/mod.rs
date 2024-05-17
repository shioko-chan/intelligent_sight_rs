mod err_code;
use crate::cuda_op::err_code::CUDA_ERR_NAME;
use crate::{UnifiedItem, UnifiedTrait};
use anyhow::{anyhow, Result};
use err_code::TRT_ERR_NAME;

mod trt_op_ffi {
    use std::ffi::c_char;
    extern "C" {
        pub fn create_engine(
            engine_filename: *const c_char,
            input_name: *const c_char,
            output_name: *const c_char,
            width: u32,
            height: u32,
        ) -> u16;
        pub fn create_context() -> u16;
        pub fn infer() -> u16;
        pub fn release_resources() -> u16;
        pub fn set_input(input_buffer: *mut f32) -> u16;
        pub fn set_output(output_buffer: *mut f32) -> u16;
    }
}

#[inline]
fn exec_and_check(mut f: impl FnMut() -> Result<u16>) -> Result<()> {
    match f()? {
        0 => Ok(()),
        err @ 1..=9999 => Err(anyhow!(
            "TRT: Failed, cuda error code: {} ({})",
            err,
            CUDA_ERR_NAME
                .get(err as usize)
                .unwrap_or(&"err code unknown")
        )),
        err => Err(anyhow!(
            "TRT: Failed, error code: {} ({})",
            err,
            TRT_ERR_NAME
                .get(err as usize)
                .unwrap_or(&"err code unknown")
        )),
    }
}

pub fn create_engine(
    engine_filename: &str,
    input_name: &str,
    output_name: &str,
    width: u32,
    height: u32,
) -> Result<()> {
    let engine_filename = std::ffi::CString::new(engine_filename)?;
    let input_name = std::ffi::CString::new(input_name)?;
    let output_name = std::ffi::CString::new(output_name)?;
    exec_and_check(|| {
        Ok(unsafe {
            trt_op_ffi::create_engine(
                engine_filename.as_ptr(),
                input_name.as_ptr(),
                output_name.as_ptr(),
                width,
                height,
            )
        })
    })
}

pub fn create_context() -> Result<()> {
    exec_and_check(|| Ok(unsafe { trt_op_ffi::create_context() }))
}

pub fn infer() -> Result<()> {
    exec_and_check(|| Ok(unsafe { trt_op_ffi::infer() }))
}

pub fn release_resources() -> Result<()> {
    exec_and_check(|| Ok(unsafe { trt_op_ffi::release_resources() }))
}

pub fn set_input(input_buffer: &mut UnifiedItem<f32>) -> Result<()> {
    exec_and_check(|| Ok(unsafe { trt_op_ffi::set_input(input_buffer.to_device()?) }))
}

pub fn set_output(output_buffer: &mut UnifiedItem<f32>) -> Result<()> {
    exec_and_check(|| Ok(unsafe { trt_op_ffi::set_output(output_buffer.device()?) }))
}

#[cfg(test)]
mod test {
    use crate::Tensor;

    use super::*;

    #[test]
    fn test_infer() {
        create_engine("../model.trt", "images", "output0", 640, 640).unwrap();
        create_context().unwrap();

        let mut input = Tensor::new(vec![1, 3, 640, 640]).unwrap();
        let mut output = Tensor::new(vec![1, 31, 8400]).unwrap();
        set_input(&mut input).unwrap();
        set_output(&mut output).unwrap();

        infer().unwrap();

        release_resources().unwrap();
        // output.to_host().unwrap();

        // for num in output.iter() {
        // println!("{:?}", num);
        // }
    }
}
