use anyhow::Result;
use std::ops::{Deref, DerefMut};

pub struct HostItem<T>(HostArray<T>);

impl<T> HostItem<T> {
    pub fn new(size: usize) -> Result<Self>
    where
        T: Default + Copy,
    {
        Ok(HostItem(HostArray::new(size)?))
    }
}

impl<T> Deref for HostItem<T> {
    type Target = HostArray<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for HostItem<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> Clone for HostItem<T>
where
    T: Default + Copy,
{
    fn clone(&self) -> Self {
        let mut item = HostItem::new(self.0.len()).expect("Failed to allocate uniform memory");
        item.iter_mut().zip(self.iter()).for_each(|(dst, src)| {
            *dst = *src;
        });
        item
    }
}

pub struct HostArray<T>(Vec<T>);

// unsafe impl<T> Send for HostArray<T> {}

impl<T> Deref for HostArray<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for HostArray<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> HostArray<T> {
    pub fn new(size: usize) -> Result<Self>
    where
        T: Default + Copy,
    {
        Ok(HostArray(vec![T::default(); size]))
    }
    pub fn from_raw_parts(ptr: *mut T, size: usize) -> Self {
        HostArray(unsafe { Vec::from_raw_parts(ptr, size, size) })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_unified_item_create() {
        let mut item: HostItem<f64> = HostItem::new(10).unwrap();
        item.iter_mut().for_each(|num| *num = 1.0);
        item.iter().for_each(|num| assert_eq!(*num, 1.0));
    }

    #[test]
    fn test_unified_item_clone() {
        let mut item: HostItem<f64> = HostItem::new(10).unwrap();
        item.iter_mut().for_each(|num| *num = 1.0);
        let item2 = item.clone();
        item2.iter().for_each(|num| assert_eq!(*num, 1.0));
    }
}
