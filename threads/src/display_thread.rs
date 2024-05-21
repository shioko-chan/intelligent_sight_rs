use anyhow::anyhow;
use cv::core::{Rect_, VecN};
use cv::types::{VectorOfPoint2f, VectorOfPoint3f};
use intelligent_sight_lib::{ImageBuffer, TensorBuffer, UnifiedTrait};
use log::error;
use opencv::core::{Point2f, Point3f};
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
                            let conf = iter.next().unwrap();
                            let cls = iter.next().unwrap();
                            println!("{} {}", conf, cls);
                            if *cls == 0.0 || *cls == 17.0 {
                                cv::imgproc::circle(
                                    &mut mat,
                                    cv::core::Point_::new(
                                        x.round() as i32,
                                        (y.round() - 80.0) as i32,
                                    ),
                                    5,
                                    VecN::new(255.0, 255.0, 255.0, 255.0),
                                    -1,
                                    0,
                                    0,
                                )
                                .unwrap();
                                cv::imgproc::rectangle(
                                    &mut mat,
                                    Rect_::new(
                                        (x - w / 2.0).round() as i32,
                                        (y - 80.0 - h / 2.0).round() as i32,
                                        w.round() as i32,
                                        h.round() as i32,
                                    ),
                                    VecN::new(255.0, 255.0, 255.0, 255.0),
                                    2,
                                    0,
                                    0,
                                )
                                .unwrap();
                            }
                            let mut image_points = Vec::with_capacity(10);
                            for _ in 0..5 {
                                let x = iter.next().unwrap();
                                let y = iter.next().unwrap();
                                image_points.push(Point2f::new(*x, *y));
                            }

                            // 准备3D点（物体坐标系）
                            let object_points: Vec<Point3f> = vec![
                                Point3f::new(0.0, 0.0, 0.0),
                                Point3f::new(1.0, 0.0, 0.0),
                                Point3f::new(0.0, 1.0, 0.0),
                                Point3f::new(1.0, 1.0, 0.0),
                            ];

                            // 定义相机内参矩阵
                            let camera_matrix = Mat::from_slice_2d(&[
                                [800.0, 0.0, 320.0],
                                [0.0, 800.0, 240.0],
                                [0.0, 0.0, 1.0],
                            ])
                            .unwrap();

                            // 畸变系数（假设无畸变）
                            let dist_coeffs = Mat::zeros(4, 1, cv::core::CV_64F).unwrap();

                            // 旋转向量和平移向量
                            let mut rvec = Mat::default();
                            let mut tvec = Mat::default();

                            cv::calib3d::solve_pnp(
                                &VectorOfPoint3f::from(object_points),
                                &VectorOfPoint2f::from(image_points),
                                &camera_matrix,
                                &dist_coeffs,
                                &mut rvec,
                                &mut tvec,
                                false,
                                cv::calib3d::SOLVEPNP_ITERATIVE,
                            )
                            .unwrap();
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
