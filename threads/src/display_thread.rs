use cv::core::{Rect_, VecN};
use intelligent_sight_lib::{ImageBuffer, TensorBuffer, UnifiedTrait};
use opencv::{self as cv, prelude::*};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    thread::{self, JoinHandle},
};

pub struct DisplayThread {
    image_rx: mpsc::Receiver<ImageBuffer>,
    detection_rx: mpsc::Receiver<TensorBuffer>,
    stop_sig: Arc<AtomicBool>,
}

impl DisplayThread {
    pub fn new(
        image_rx: mpsc::Receiver<ImageBuffer>,
        detection_rx: mpsc::Receiver<TensorBuffer>,
        stop_sig: Arc<AtomicBool>,
    ) -> Self {
        DisplayThread {
            image_rx,
            detection_rx,
            stop_sig,
        }
    }

    pub fn run(self) -> JoinHandle<()> {
        thread::spawn(move || {
            while self.stop_sig.load(Ordering::Relaxed) == false {
                if let Ok(mut image) = self.image_rx.try_recv() {
                    if let Ok(detection) = self.detection_rx.try_recv() {
                        let mut mat = unsafe {
                            Mat::new_rows_cols_with_data_unsafe(
                                image.height as i32,
                                image.width as i32,
                                cv::core::CV_8UC3,
                                image.host() as *mut std::ffi::c_void,
                                image.width as usize * 3 * std::mem::size_of::<u8>(),
                            )
                            .unwrap()
                        };
                        cv::highgui::imshow("Display", &mat).unwrap();
                        let mut iter = detection.iter();
                        for _ in 0..detection.size()[0] {
                            let x = iter.next().unwrap();
                            let y = iter.next().unwrap();
                            let w = iter.next().unwrap();
                            let h = iter.next().unwrap();
                            cv::imgproc::rectangle(
                                &mut mat,
                                Rect_::new(
                                    x.round() as i32,
                                    y.round() as i32,
                                    w.round() as i32,
                                    h.round() as i32,
                                ),
                                VecN::new(255.0, 255.0, 255.0, 255.0),
                                2,
                                cv::core::LINE_8,
                                0,
                            )
                            .unwrap();
                            for _ in 0..12 {
                                iter.next();
                            }
                        }
                        cv::highgui::wait_key(1).unwrap();
                    }
                }
            }
            self.stop_sig.store(true, Ordering::Relaxed);
        })
    }
}
