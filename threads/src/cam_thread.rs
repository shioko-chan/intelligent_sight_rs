use crate::thread_trait::Processor;
use anyhow::{anyhow, Result};
use intelligent_sight_lib::{
    get_image, initialize_camera, uninitialize_camera, FlipFlag, ImageBuffer, Reader, Writer,
};
use log::{debug, error, info, log_enabled};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

pub struct CamThread {
    shared_buffer: Writer<ImageBuffer>,
    stop_sig: Arc<AtomicBool>,
}

impl CamThread {
    pub fn new(stop_sig: Arc<AtomicBool>) -> Result<Self> {
        let mut buffer_width = vec![0u32; 1];
        let mut buffer_height = vec![0u32; 1];
        let mut initialize_retry = 0;

        while let Err(err) = initialize_camera(1, &mut buffer_width, &mut buffer_height) {
            error!(
                "CamThread: Failed to initialize camera with err: {}, retrying...",
                err
            );
            initialize_retry += 1;
            if initialize_retry > 10 {
                error!("CamThread: Failed to initialize camera after 10 retries, exiting...");
                return Err(anyhow!("CamThread: Failed to initialize camera {}", err));
            }
        }
        info!(
            "CamThread: Camera initialized, width: {}, height: {}",
            buffer_width[0], buffer_height[0]
        );
        Ok(CamThread {
            shared_buffer: Writer::new(4, || ImageBuffer::new(buffer_width[0], buffer_height[0]))?,
            stop_sig,
        })
    }
}

impl Drop for CamThread {
    fn drop(&mut self) {
        let mut uninitialize_retry = 0;
        while let Err(err) = uninitialize_camera() {
            error!(
                "CamThread: Failed to uninitialize camera with err: {}, retrying...",
                err
            );
            uninitialize_retry += 1;
            if uninitialize_retry > 10 {
                error!("CamThread: Failed to uninitialize camera after 10 retries, exiting...");
            }
        }
    }
}

impl Processor for CamThread {
    type Output = ImageBuffer;

    fn get_output_buffer(&self) -> Reader<ImageBuffer> {
        self.shared_buffer.get_reader()
    }

    fn start_processor(self) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            let mut cnt = 0;
            let mut start = std::time::Instant::now();

            while self.stop_sig.load(Ordering::Relaxed) == false {
                let mut lock = self.shared_buffer.write();

                if let Err(err) = get_image(0, &mut lock, FlipFlag::None) {
                    error!("err: {}", err);
                    break;
                }

                drop(lock);

                cnt += 1;
                if cnt == 10 {
                    let end = std::time::Instant::now();
                    let elapsed = end.duration_since(start);
                    if log_enabled!(log::Level::Debug) {
                        debug!("CamThread: fps: {}", 10.0 / elapsed.as_secs_f32());
                    }
                    start = end;
                    cnt = 0;
                }
            }
            self.stop_sig.store(true, Ordering::Relaxed);
        })
    }
}
