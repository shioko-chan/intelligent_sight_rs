mod err_code;
use crate::{unified_item::CArray, Image, Tensor};
use anyhow::{anyhow, Result};
use err_code::CUDA_ERR_NAME;
use std::mem;

mod cuda_op_ffi {
    extern "C" {
        pub fn cuda_malloc(size: u32, ptr: *mut *mut u8) -> u16;
        pub fn cuda_free(ptr: *mut u8) -> u16;
        pub fn convert_rgb888_3dtensor(
            input_buffer: *const u8,
            output_buffer: *mut f32,
            width: u32,
            height: u32,
        ) -> u16;

    }
}

pub fn cuda_malloc<T>(size: usize) -> Result<CArray<T>>
where
    T: Sized,
{
    let mut ptr = std::ptr::null_mut();
    match unsafe {
        cuda_op_ffi::cuda_malloc(
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
    match unsafe { cuda_op_ffi::cuda_free(ptr as *mut u8) } {
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

pub fn convert_rgb888_3dtensor(input_image: &Image, output_tensor: &mut Tensor) -> Result<()> {
    match unsafe {
        cuda_op_ffi::convert_rgb888_3dtensor(
            input_image.data.as_ptr(),
            output_tensor.data.as_mut_ptr(),
            input_image.width,
            input_image.height,
        )
    } {
        0 => Ok(()),
        err => Err(anyhow!(
            "Failed to convert rgb888 to 3d tensor, code: {} ({})",
            err,
            CUDA_ERR_NAME
                .get(err as usize)
                .unwrap_or(&"err code unknown")
        )),
    }
}
