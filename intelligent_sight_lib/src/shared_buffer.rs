use anyhow::Result;
use std::{
    ops::{Deref, DerefMut},
    sync::{Mutex, MutexGuard},
};

pub struct SharedBufferLock<'a, T> {
    _id: usize,
    lock: MutexGuard<'a, T>,
    is_read: bool,
    shared_buffer: &'a SharedBuffer<T>,
}

impl<T> Deref for SharedBufferLock<'_, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &*self.lock
    }
}

impl<T> DerefMut for SharedBufferLock<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.lock
    }
}

impl<T> Drop for SharedBufferLock<'_, T> {
    fn drop(&mut self) {
        if self.is_read {
            self.shared_buffer.read_finish(self._id);
        } else {
            self.shared_buffer.write_finish(self._id);
        }
    }
}

pub struct SharedBuffer<T> {
    info: Mutex<Vec<BufferInfo>>,
    buffers: Vec<Mutex<T>>,
}

struct BufferInfo {
    lfu: usize,
    occupied: bool,
}

impl Default for BufferInfo {
    fn default() -> Self {
        BufferInfo {
            lfu: 0,
            occupied: false,
        }
    }
}

impl Clone for BufferInfo {
    fn clone(&self) -> Self {
        BufferInfo {
            lfu: self.lfu,
            occupied: self.occupied,
        }
    }
}

impl Copy for BufferInfo {}

impl<T> SharedBuffer<T> {
    pub fn new(reader_writer_cnt: usize, f: impl Fn() -> Result<T>) -> Result<Self> {
        let mut vec = Vec::with_capacity(reader_writer_cnt);
        for _ in 0..reader_writer_cnt {
            vec.push(Mutex::new(f()?));
        }
        Ok(SharedBuffer {
            info: Mutex::new(vec![BufferInfo::default(); reader_writer_cnt]),
            buffers: vec,
        })
    }

    #[inline]
    fn get_buffer_info(&self) -> MutexGuard<Vec<BufferInfo>> {
        match self.info.lock() {
            Ok(info) => info,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    #[inline]
    fn get_read_index(&self) -> usize {
        let mut info = self.get_buffer_info();
        let index = info
            .iter()
            .enumerate()
            .filter(|x| !x.1.occupied)
            .min_by_key(|x| x.1.lfu)
            .unwrap()
            .0;
        info[index].occupied = true;
        index
    }

    #[inline]
    fn get_write_index(&self) -> usize {
        let mut info = self.get_buffer_info();
        let index = info
            .iter()
            .enumerate()
            .filter(|x| !x.1.occupied)
            .max_by_key(|x| x.1.lfu)
            .unwrap()
            .0;
        info[index].occupied = true;
        index
    }

    #[inline]
    fn get_buffer(&self, index: usize) -> MutexGuard<T> {
        match self.buffers[index].lock() {
            Ok(buffer) => buffer,
            Err(poisoned) => poisoned.into_inner(),
        }
    }

    pub fn read(&self) -> SharedBufferLock<T> {
        let index = self.get_read_index();
        SharedBufferLock {
            _id: index,
            is_read: true,
            lock: self.get_buffer(index),
            shared_buffer: self,
        }
    }

    fn read_finish(&self, id: usize) {
        let mut info = self.get_buffer_info();
        info[id].occupied = false;
    }

    pub fn write(&self) -> SharedBufferLock<T> {
        let index = self.get_write_index();
        SharedBufferLock {
            _id: index,
            is_read: false,
            lock: self.get_buffer(index),
            shared_buffer: self,
        }
    }

    fn write_finish(&self, id: usize) {
        let mut info = self.get_buffer_info();
        info.iter_mut().for_each(|x| x.lfu += 1);
        info[id].lfu = 0;
        info[id].occupied = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;
    #[test]
    fn test_shared_buffer_spsc() {
        let n = 1000000;
        let buffer = SharedBuffer::new(10, || Ok(0usize)).unwrap();
        let share_buffer1 = Arc::new(buffer);
        let share_buffer2 = share_buffer1.clone();
        let buffer = share_buffer2.clone();
        let handle1 = thread::spawn(move || {
            for _ in 0..n {
                #[allow(unused)]
                let read_buffer = share_buffer1.read();
            }
        });
        let handle2 = thread::spawn(move || {
            for i in 0..n {
                let mut write_buffer = share_buffer2.write();
                *write_buffer = i;
            }
        });
        handle1.join().unwrap();
        handle2.join().unwrap();
        assert_eq!(n - 1, *buffer.read().lock);
    }

    #[test]
    fn test_shared_buffer_mpsc() {
        let n = 1000000;
        let buffer = SharedBuffer::new(10, || Ok(0usize)).unwrap();
        let share_buffer1 = Arc::new(buffer);
        let share_buffer2 = share_buffer1.clone();
        let share_buffer3 = share_buffer1.clone();
        let buffer = share_buffer1.clone();
        let handle1 = thread::spawn(move || {
            for _ in 0..n {
                #[allow(unused)]
                let read_buffer = share_buffer1.read();
            }
        });
        let handle2 = thread::spawn(move || {
            for i in 0..n {
                let mut write_buffer = share_buffer2.write();
                *write_buffer = i;
            }
        });
        let handle3 = thread::spawn(move || {
            for i in 1..n + 1 {
                let mut write_buffer = share_buffer3.write();
                *write_buffer = i;
            }
        });
        handle1.join().unwrap();
        handle2.join().unwrap();
        handle3.join().unwrap();
        let latest = buffer.read();
        assert!(n - 1 == *latest.lock || n == *latest.lock);
    }
}
