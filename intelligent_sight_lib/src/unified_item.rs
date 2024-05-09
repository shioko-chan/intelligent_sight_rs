use crate::trt_op::{cuda_free, cuda_malloc};
use anyhow::{anyhow, Result};
use std::ops::{Deref, DerefMut};
pub struct CArray<T> {
    ptr: *mut T,
    size: usize,
}

pub struct CArrayIter<'a, T> {
    ptr: *mut T,
    index: usize,
    size: usize,
    _marker: std::marker::PhantomData<&'a T>,
}

impl<'a, T> Iterator for CArrayIter<'a, T> {
    type Item = &'a T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.size {
            return None;
        }
        let ret = unsafe { self.ptr.add(self.index).as_ref() };
        self.index += 1;
        ret
    }
}

pub struct CArrayIterMut<'a, T> {
    ptr: *mut T,
    index: usize,
    size: usize,
    _marker: std::marker::PhantomData<&'a T>,
}

impl<'a, T> Iterator for CArrayIterMut<'a, T> {
    type Item = &'a mut T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.size {
            return None;
        }
        let ret = unsafe { self.ptr.add(self.index).as_mut() };
        self.index += 1;
        ret
    }
}

unsafe impl<T> Send for CArray<T> {}

impl<T> Clone for CArray<T> {
    fn clone(&self) -> Self {
        CArray {
            ptr: self.ptr,
            size: self.size,
        }
    }
}

impl<T> CArray<T> {
    pub fn from_raw_parts(ptr: *mut T, size: usize) -> Self {
        CArray { ptr, size }
    }

    pub fn len(&self) -> usize {
        self.size
    }

    pub fn iter(&self) -> CArrayIter<'_, T> {
        CArrayIter {
            ptr: self.ptr,
            index: 0,
            size: self.size,
            _marker: std::marker::PhantomData,
        }
    }

    pub fn iter_mut(&mut self) -> CArrayIterMut<'_, T> {
        CArrayIterMut {
            ptr: self.ptr,
            index: 0,
            size: self.size,
            _marker: std::marker::PhantomData,
        }
    }

    pub unsafe fn as_ptr(&self) -> *const T {
        self.ptr as *const T
    }

    pub unsafe fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

pub struct UnifiedItem<T>(CArray<T>);

impl<T> UnifiedItem<T> {
    pub fn new(size: usize) -> Result<Self> {
        Ok(UnifiedItem(cuda_malloc(size)?))
    }
}

impl<T> Deref for UnifiedItem<T> {
    type Target = CArray<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for UnifiedItem<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> Drop for UnifiedItem<T> {
    fn drop(&mut self) {
        cuda_free(&mut self.0).expect("Failed to free uniform memory");
    }
}

impl<T> Clone for UnifiedItem<T>
where
    T: Copy,
{
    fn clone(&self) -> Self {
        let mut item = UnifiedItem::new(self.0.len()).expect("Failed to allocate uniform memory");
        item.iter_mut().zip(self.iter()).for_each(|(dst, src)| {
            *dst = *src;
        });
        item
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_unified_item_create() {
        let mut item: UnifiedItem<f64> = UnifiedItem::new(10).unwrap();
        item.iter_mut().for_each(|num| *num = 1.0);
        item.iter().for_each(|num| assert_eq!(*num, 1.0));
    }

    #[test]
    fn test_unified_item_clone() {
        let mut item: UnifiedItem<f64> = UnifiedItem::new(10).unwrap();
        item.iter_mut().for_each(|num| *num = 1.0);
        let item2 = item.clone();
        item2.iter().for_each(|num| assert_eq!(*num, 1.0));
    }
}
