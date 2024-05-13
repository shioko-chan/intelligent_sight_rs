pub(crate) mod err_code;
use crate::{device_item::DeviceArray, unified_item::ManagedArray, Image, Tensor};
use anyhow::{anyhow, Result};
use err_code::CUDA_ERR_NAME;
use std::mem;

mod cuda_op_ffi {
    extern "C" {
        pub fn cuda_malloc(size: u32, ptr: *mut *mut u8) -> u16;
        pub fn cuda_malloc_managed(size: u32, ptr: *mut *mut u8) -> u16;
        pub fn cuda_free(ptr: *mut u8) -> u16;
        pub fn transfer_host_to_device(
            host_buffer: *const u8,
            device_buffer: *mut u8,
            size: u32,
        ) -> u16;
        pub fn transfer_device_to_host(
            host_buffer: *const u8,
            device_buffer: *mut u8,
            size: u32,
        ) -> u16;
        pub fn convert_rgb888_3dtensor(
            input_buffer: *const u8,
            output_buffer: *mut f32,
            width: u32,
            height: u32,
        ) -> u16;
    }
}

pub fn cuda_malloc<T>(size: usize) -> Result<*mut T>
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
        0 => Ok(ptr),
        err => Err(anyhow!(
            "Failed to allocate memory, code: {} ({})",
            err,
            CUDA_ERR_NAME
                .get(err as usize)
                .unwrap_or(&"err code unknown")
        )),
    }
}

pub fn cuda_malloc_managed<T>(size: usize) -> Result<ManagedArray<T>>
where
    T: Sized,
{
    let mut ptr = std::ptr::null_mut();
    match unsafe {
        cuda_op_ffi::cuda_malloc_managed(
            (size * mem::size_of::<T>() / mem::size_of::<u8>()) as u32,
            &mut ptr as *mut *mut T as *mut *mut u8,
        )
    } {
        0 => Ok(ManagedArray::from_raw_parts(ptr, size)),
        err => Err(anyhow!(
            "Failed to allocate memory, code: {} ({})",
            err,
            CUDA_ERR_NAME
                .get(err as usize)
                .unwrap_or(&"err code unknown")
        )),
    }
}

pub fn cuda_free<T>(vec: &mut ManagedArray<T>) -> Result<()> {
    let ptr = vec.as_mut_ptr();
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
            input_image.as_ptr(),
            output_tensor.as_mut_ptr(),
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

pub fn transfer_host_to_device<T>(
    host_buffer: &ManagedArray<T>,
    output_buffer: &mut DeviceArray<T>,
    size: usize,
) -> Result<()> {
    match unsafe {
        cuda_op_ffi::transfer_host_to_device(
            host_buffer.as_ptr() as *const u8,
            output_buffer.as_mut_ptr() as *mut u8,
            size as u32,
        )
    } {
        0 => Ok(()),
        err => Err(anyhow!(
            "Failed to transfer memory to device, code: {} ({})",
            err,
            CUDA_ERR_NAME
                .get(err as usize)
                .unwrap_or(&"err code unknown")
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_convert_img_tensor() {
        let image = Image::new(640, 480).unwrap();
        let mut tensor = Tensor::new(vec![640, 640, 3]).unwrap();
        convert_rgb888_3dtensor(&image, &mut tensor).unwrap();
        for data in tensor.iter().take(640 * 80) {
            assert_eq!(*data, 0.5);
        }
    }
}
