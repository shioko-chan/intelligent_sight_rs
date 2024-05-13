use anyhow::Result;
use std::ops::{Deref, DerefMut};

#[cfg(target_os = "windows")]
pub use unified_item::*;

#[cfg(target_os = "linux")]
pub use unified_item_cuda::*;

pub struct UnifiedItem<T>(ManagedArray<T>);

impl<T> UnifiedItem<T> {
    pub fn new(size: usize) -> Result<Self>
    where
        T: Default + Copy,
    {
        Ok(UnifiedItem(ManagedArray::new(size)?))
    }
}

impl<T> Deref for UnifiedItem<T> {
    type Target = ManagedArray<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for UnifiedItem<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> Clone for UnifiedItem<T>
where
    T: Default + Copy,
{
    fn clone(&self) -> Self {
        let mut item = UnifiedItem::new(self.0.len()).expect("Failed to allocate uniform memory");
        item.iter_mut().zip(self.iter()).for_each(|(dst, src)| {
            *dst = *src;
        });
        item
    }
}

#[cfg(target_os = "windows")]
mod unified_item {
    use anyhow::Result;
    use std::ops::{Deref, DerefMut};

    pub struct ManagedArray<T>(Vec<T>);

    unsafe impl<T> Send for ManagedArray<T> {}

    impl<T> Deref for ManagedArray<T> {
        type Target = Vec<T>;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl<T> DerefMut for ManagedArray<T> {
        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    impl<T> ManagedArray<T> {
        pub fn new(size: usize) -> Result<Self>
        where
            T: Default + Copy,
        {
            Ok(ManagedArray(vec![T::default(); size]))
        }
        pub fn from_raw_parts(ptr: *mut T, size: usize) -> Self {
            ManagedArray(unsafe { Vec::from_raw_parts(ptr, size, size) })
        }
    }
}

#[cfg(target_os = "linux")]
mod unified_item_cuda {
    use crate::{cuda_free, cuda_malloc_managed};

    use anyhow::Result;

    pub struct ManagedArrayIter<'a, T> {
        ptr: *const T,
        index: usize,
        size: usize,
        _marker: std::marker::PhantomData<&'a T>,
    }

    impl<'a, T> Iterator for ManagedArrayIter<'a, T> {
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

    pub struct ManagedArrayIterMut<'a, T> {
        ptr: *mut T,
        index: usize,
        size: usize,
        _marker: std::marker::PhantomData<&'a T>,
    }

    impl<'a, T> Iterator for ManagedArrayIterMut<'a, T> {
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

    pub struct ManagedArray<T> {
        ptr: *mut T,
        size: usize,
    }

    unsafe impl<T> Send for ManagedArray<T> {}

    impl<T> Clone for ManagedArray<T> {
        fn clone(&self) -> Self {
            ManagedArray {
                ptr: self.ptr,
                size: self.size,
            }
        }
    }

    impl<T> Drop for ManagedArray<T> {
        fn drop(&mut self) {
            cuda_free(self).expect("Failed to free uniform memory");
        }
    }

    impl<T> ManagedArray<T> {
        pub fn new(size: usize) -> Result<Self> {
            cuda_malloc_managed(size)
        }
        pub fn from_raw_parts(ptr: *mut T, size: usize) -> Self {
            ManagedArray { ptr, size }
        }

        pub fn len(&self) -> usize {
            self.size
        }

        pub fn iter(&self) -> ManagedArrayIter<'_, T> {
            ManagedArrayIter {
                ptr: self.ptr as *const T,
                index: 0,
                size: self.size,
                _marker: std::marker::PhantomData,
            }
        }

        pub fn iter_mut(&mut self) -> ManagedArrayIterMut<'_, T> {
            ManagedArrayIterMut {
                ptr: self.ptr,
                index: 0,
                size: self.size,
                _marker: std::marker::PhantomData,
            }
        }

        pub fn as_ptr(&self) -> *const T {
            self.ptr as *const T
        }

        pub fn as_mut_ptr(&mut self) -> *mut T {
            self.ptr
        }
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
