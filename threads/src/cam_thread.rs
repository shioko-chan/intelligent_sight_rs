use crate::thread_trait::Processor;
use anyhow::{anyhow, Result};

use intelligent_sight_lib::{
    get_image, initialize_camera, uninitialize_camera, FlipFlag, SharedBuffer, UnifiedTrait,
};
use log::{debug, info, log_enabled, warn};
use opencv::{self as cv, prelude::*};
use std::os::raw::c_void;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

pub struct CamThread {
    shared_buffer: Arc<SharedBuffer<intelligent_sight_lib::Image>>,
    stop_sig: Arc<AtomicBool>,
}

impl CamThread {
    pub fn new(stop_sig: Arc<AtomicBool>) -> Result<Self> {
        let mut buffer_width = vec![0u32; 1];
        let mut buffer_height = vec![0u32; 1];
        let mut initialize_retry = 0;

        while let Err(err) = initialize_camera(1, &mut buffer_width, &mut buffer_height) {
            warn!(
                "CamThread: Failed to initialize camera with err: {}, retrying...",
                err
            );
            initialize_retry += 1;
            if initialize_retry > 10 {
                warn!("CamThread: Failed to initialize camera after 10 retries, exiting...");
                return Err(anyhow!("CamThread: Failed to initialize camera {}", err));
            }
        }
        info!(
            "CamThread: Camera initialized, width: {}, height: {}",
            buffer_width[0], buffer_height[0]
        );
        Ok(CamThread {
            shared_buffer: Arc::new(SharedBuffer::new(4, || {
                intelligent_sight_lib::Image::new(buffer_width[0], buffer_height[0])
            })?),
            stop_sig,
        })
    }
}

impl Drop for CamThread {
    fn drop(&mut self) {
        let mut uninitialize_retry = 0;
        while let Err(err) = uninitialize_camera() {
            warn!(
                "CamThread: Failed to uninitialize camera with err: {}, retrying...",
                err
            );
            uninitialize_retry += 1;
            if uninitialize_retry > 10 {
                warn!("CamThread: Failed to uninitialize camera after 10 retries, exiting...");
            }
        }
    }
}

impl Processor for CamThread {
    type Output = intelligent_sight_lib::Image;

    fn get_output_buffer(&self) -> Arc<SharedBuffer<intelligent_sight_lib::Image>> {
        self.shared_buffer.clone()
    }

    fn start_processor(&self) -> thread::JoinHandle<()> {
        let stop_sig = self.stop_sig.clone();
        let shared_buffer = self.shared_buffer.clone();

        thread::spawn(move || {
            let mut cnt = 0;
            let mut start = std::time::Instant::now();

            cv::highgui::named_window("CamThread", cv::highgui::WINDOW_AUTOSIZE).unwrap();
            while stop_sig.load(Ordering::Relaxed) == false {
                let mut lock = shared_buffer.write();

                if let Err(err) = get_image(0, &mut lock, FlipFlag::None) {
                    warn!("err: {}", err);
                    break;
                }

                let mat = unsafe {
                    Mat::new_rows_cols_with_data_unsafe(
                        lock.height as i32,
                        lock.width as i32,
                        cv::core::CV_8UC3,
                        lock.host() as *mut c_void,
                        lock.width as usize * 3 * std::mem::size_of::<u8>(),
                    )
                    .unwrap()
                };

                cv::highgui::imshow("CamThread", &mat).unwrap();
                let ret = cv::highgui::wait_key(1).unwrap();
                if ret == 'q' as i32 {
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
            stop_sig.store(true, Ordering::Relaxed);
        })
    }
}
