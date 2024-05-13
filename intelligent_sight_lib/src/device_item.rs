use crate::cuda_malloc;
use anyhow::Result;
use std::ops::{Deref, DerefMut};

#[derive(Clone)]
pub struct DeviceArray<T> {
    pub size: usize,
    data: *mut T,
}

impl<T> Deref for DeviceArray<T> {
    type Target = *mut T;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> DerefMut for DeviceArray<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<T> DeviceArray<T> {
    pub fn new(size: usize) -> Result<Self> {
        Ok(DeviceArray {
            data: cuda_malloc(size)?,
            size,
        })
    }

    pub fn as_ptr(&self) -> *const T {
        self.data
    }

    pub fn as_mut_ptr(&self) -> *mut T {
        self.data
    }
}
