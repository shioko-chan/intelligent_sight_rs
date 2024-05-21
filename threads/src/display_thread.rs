use anyhow::anyhow;
use cv::core::{Rect_, VecN, LINE_8};
use intelligent_sight_lib::{ImageBuffer, TensorBuffer, UnifiedTrait};
use log::error;
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
                if let Ok(detection) = self.detection_rx.recv() {
                    let get_image = || loop {
                        match self.image_rx.recv() {
                            Ok(image) => {
                                if image.timestamp == detection.timestamp {
                                    return Ok(image);
                                }
                            }
                            Err(err) => {
                                if self.stop_sig.load(Ordering::Relaxed) == false {
                                    error!("DisplayThread: Failed to get image: {}", err);
                                }
                                return Err(anyhow!("DisplayThread: Failed to get image: {}", err));
                            }
                        }
                    };

                    if let Ok(mut image) = get_image() {
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
                        let mut iter = detection.iter();

                        for _ in 0..detection.size()[0] {
                            let x = iter.next().unwrap();
                            let y = iter.next().unwrap();
                            let w = iter.next().unwrap();
                            let h = iter.next().unwrap();
                            println!("{} {} {} {}", x, y, w, h);
                            println!("{} {}", iter.next().unwrap(), iter.next().unwrap());
                            cv::imgproc::circle(
                                &mut mat,
                                cv::core::Point_::new((x.round() - 80.0) as i32, y.round() as i32),
                                5,
                                VecN::new(255.0, 255.0, 1.0, 1.0),
                                -1,
                                LINE_8,
                                0,
                            )
                            .unwrap();
                            cv::imgproc::rectangle(
                                &mut mat,
                                Rect_::new(
                                    (x - w / 2.0 - 80.0).round() as i32,
                                    (y - h / 2.0).round() as i32,
                                    w.round() as i32,
                                    h.round() as i32,
                                ),
                                VecN::new(255.0, 255.0, 255.0, 255.0),
                                2,
                                cv::core::LINE_8,
                                0,
                            )
                            .unwrap();
                            for _ in 0..10 {
                                iter.next();
                            }
                        }
                        cv::highgui::imshow("Display", &mat).unwrap();
                        cv::highgui::wait_key(1).unwrap();
                    }
                }
            }
            cv::highgui::destroy_all_windows().unwrap();
            self.stop_sig.store(true, Ordering::Relaxed);
        })
    }
}
