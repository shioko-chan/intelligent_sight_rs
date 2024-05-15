pub(crate) mod err_code;
use crate::{Image, Tensor, UnifiedTrait};
use anyhow::{anyhow, Result};
use err_code::CUDA_ERR_NAME;
use std::mem;

mod cuda_op_ffi {
    extern "C" {
        pub fn cuda_malloc(size: u32, ptr: *mut *mut u8) -> u16;
        pub fn cuda_malloc_managed(size: u32, ptr: *mut *mut u8) -> u16;
        pub fn cuda_malloc_host(size: u32, ptr: *mut *mut u8) -> u16;
        pub fn cuda_free(ptr: *mut u8) -> u16;
        pub fn cuda_free_host(ptr: *mut u8) -> u16;
        pub fn transfer_host_to_device(
            host_buffer: *const u8,
            device_buffer: *mut u8,
            size: u32,
        ) -> u16;
        pub fn transfer_device_to_host(
            host_buffer: *mut u8,
            device_buffer: *const u8,
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

#[inline]
fn exec_and_check(function: impl FnOnce() -> Result<u16>) -> Result<()> {
    match function()? {
        0 => Ok(()),
        err_code => Err(anyhow!(
            "operation failed, code: {} ({})",
            err_code,
            CUDA_ERR_NAME
                .get(err_code as usize)
                .unwrap_or(&"err code unknown")
        )),
    }
}

pub fn cuda_malloc<T>(size: usize) -> Result<*mut T>
where
    T: Sized,
{
    let mut ptr = std::ptr::null_mut();
    exec_and_check(|| {
        Ok(unsafe {
            cuda_op_ffi::cuda_malloc(
                (size * mem::size_of::<T>() / mem::size_of::<u8>()) as u32,
                &mut ptr as *mut *mut T as *mut *mut u8,
            )
        })
    })
    .map(|_| ptr)
}

pub fn cuda_malloc_managed<T>(size: usize) -> Result<*mut T>
where
    T: Sized,
{
    let mut ptr = std::ptr::null_mut();
    exec_and_check(|| {
        Ok(unsafe {
            cuda_op_ffi::cuda_malloc_managed(
                (size * mem::size_of::<T>() / mem::size_of::<u8>()) as u32,
                &mut ptr as *mut *mut T as *mut *mut u8,
            )
        })
    })
    .map(|_| ptr)
}

pub fn cuda_malloc_host<T>(size: usize) -> Result<*mut T>
where
    T: Sized,
{
    let mut ptr = std::ptr::null_mut();
    exec_and_check(|| {
        Ok(unsafe {
            cuda_op_ffi::cuda_malloc_host(
                (size * mem::size_of::<T>() / mem::size_of::<u8>()) as u32,
                &mut ptr as *mut *mut T as *mut *mut u8,
            )
        })
    })
    .map(|_| ptr)
}

pub fn cuda_free<T>(ptr: *mut T) -> Result<()> {
    exec_and_check(|| Ok(unsafe { cuda_op_ffi::cuda_free(ptr as *mut u8) }))
}

pub fn cuda_free_host<T>(ptr: *mut T) -> Result<()> {
    exec_and_check(|| Ok(unsafe { cuda_op_ffi::cuda_free_host(ptr as *mut u8) }))
}

pub fn convert_rgb888_3dtensor(input_image: &mut Image, output_tensor: &mut Tensor) -> Result<()> {
    exec_and_check(|| {
        Ok(unsafe {
            cuda_op_ffi::convert_rgb888_3dtensor(
                input_image.to_device()?,
                output_tensor.device()?,
                input_image.width,
                input_image.height,
            )
        })
    })
}

pub fn transfer_host_to_device<T>(
    host_buffer: *const T,
    device_buffer: *mut T,
    size: usize,
) -> Result<()> {
    exec_and_check(|| {
        Ok(unsafe {
            cuda_op_ffi::transfer_host_to_device(
                host_buffer as *const u8,
                device_buffer as *mut u8,
                size as u32,
            )
        })
    })
}

pub fn transfer_device_to_host<T>(
    host_buffer: *mut T,
    device_buffer: *const T,
    size: usize,
) -> Result<()> {
    exec_and_check(|| {
        Ok(unsafe {
            cuda_op_ffi::transfer_device_to_host(
                host_buffer as *mut u8,
                device_buffer as *const u8,
                size as u32,
            )
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_malloc() {
        let ptr: *mut f64 = cuda_malloc(1024).expect("malloc error");
        cuda_free(ptr).expect("free error");
    }

    #[test]
    fn test_malloc_host() {
        let ptr: *mut u128 = cuda_malloc_host(1024).expect("malloc error");
        cuda_free_host(ptr).expect("free error");
    }

    #[test]
    fn test_malloc_managed() {
        let ptr: *mut f32 = cuda_malloc_managed(1024).expect("malloc error");
        cuda_free(ptr).expect("free error");
    }

    #[test]
    fn test_convert_img_tensor() {
        let mut image = Image::new(640, 480).unwrap();
        let mut tensor = Tensor::new(vec![640, 640, 3]).unwrap();
        convert_rgb888_3dtensor(&mut image, &mut tensor).unwrap();
        for data in tensor.iter().take(640 * 80) {
            assert_eq!(*data, 0.5);
        }
    }
}
