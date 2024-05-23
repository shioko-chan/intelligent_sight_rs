use crate::thread_trait::Processor;
use anyhow::{anyhow, Result};
use intelligent_sight_lib::{Reader, TensorBuffer, UnifiedTrait, Writer};
use log::error;
use opencv::{self as cv, core::*};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc, Arc,
    },
    thread::{self, JoinHandle},
};

pub struct AnalysisThread {
    input_buffer: Reader<TensorBuffer>,
    output_buffer: Writer<Vec<f32>>,
    stop_sig: Arc<AtomicBool>,
}

impl AnalysisThread {
    const POWER_RUNE_WIDTH: f64 = 32.0;
    const POWER_RUNE_HEIGHT: f64 = 10.26;
    const CLASSES: [&'static str; 18] = [
        "PR", "B1", "B2", "B3", "B4", "B5", "BG", "BO", "BB", "R1", "R2", "R3", "R4", "R5", "RG",
        "RO", "RB", "PB",
    ];
    const POWER_RUNE_POINTS: [Point3_<f64>; 4] = [
        Point3d::new(
            Self::POWER_RUNE_WIDTH / 2.0,
            -Self::POWER_RUNE_HEIGHT / 2.0,
            0.0,
        ),
        Point3d::new(
            Self::POWER_RUNE_WIDTH / 2.0,
            Self::POWER_RUNE_HEIGHT / 2.0,
            0.0,
        ),
        Point3d::new(
            -Self::POWER_RUNE_WIDTH / 2.0,
            Self::POWER_RUNE_HEIGHT / 2.0,
            0.0,
        ),
        Point3d::new(
            -Self::POWER_RUNE_WIDTH / 2.0,
            -Self::POWER_RUNE_HEIGHT / 2.0,
            0.0,
        ),
    ];
    const COLORS: [VecN<f64, 4>; 5] = [
        VecN::new(0.0, 0.0, 255.0, 255.0),
        VecN::new(0.0, 255.0, 0.0, 255.0),
        VecN::new(255.0, 0.0, 0.0, 255.0),
        VecN::new(255.0, 255.0, 0.0, 255.0),
        VecN::new(255.0, 0.0, 255.0, 255.0),
    ];

    pub fn new(input_buffer: Reader<TensorBuffer>, stop_sig: Arc<AtomicBool>) -> Result<Self> {
        Ok(AnalysisThread {
            input_buffer,
            output_buffer: Writer::new(4, || Ok(vec![0.0; 100]))?,
            stop_sig,
        })
    }
}

impl Processor for AnalysisThread {
    type Output = Vec<f32>;

    fn get_output_buffer(&self) -> intelligent_sight_lib::Reader<Self::Output> {
        self.output_buffer.get_reader()
    }

    fn start_processor(self) -> std::thread::JoinHandle<()> {
        thread::spawn(move || {
            while self.stop_sig.load(Ordering::Relaxed) == false {
                let Some(lock_input) = self.input_buffer.read() else {
                    if self.stop_sig.load(Ordering::Relaxed) == false {
                        error!("AnalysisThread: Failed to get input");
                    }
                    break;
                };

                let mut iter = lock_input.iter();
                for _ in 0..lock_input.size()[0] {
                    let x = iter.next().unwrap();
                    let y = iter.next().unwrap();
                    let w = iter.next().unwrap();
                    let h = iter.next().unwrap();
                    // println!("{} {} {} {}", x, y, w, h);
                    let conf = iter.next().unwrap();
                    let cls = iter.next().unwrap();
                    // println!("{} {}", conf, cls);
                    if *cls != 0.0 && *cls != 17.0 {
                        continue;
                    }

                    let cls = match Self::CLASSES.get(*cls as usize) {
                        Some(cls) => cls,
                        None => {
                            self.stop_sig.store(true, Ordering::Relaxed);
                            return;
                        }
                    };

                    let mut image_points = Vector::<Point2d>::with_capacity(5);
                    for i in 0..5 {
                        let x = iter.next().unwrap();
                        let y = iter.next().unwrap();
                        if i != 2 {
                            image_points.push(Point2d::new(*x as f64, (y - 80.0) as f64));
                        }
                    }

                    // 准备3D点（物体坐标系）
                    let object_points = Vector::from_slice(&Self::POWER_RUNE_POINTS);

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
                        &object_points,
                        &image_points,
                        &camera_matrix,
                        &dist_coeffs,
                        &mut rvec,
                        &mut tvec,
                        false,
                        cv::calib3d::SOLVEPNP_ITERATIVE,
                    )
                    .unwrap();

                    let mut image_points = Vector::<Point2d>::with_capacity(4);
                    cv::calib3d::project_points_def(
                        &Vector::from_slice(&[
                            Point3d::new(0.0, 0.0, 0.0),
                            Point3d::new(20.0, 0.0, 0.0),
                            Point3d::new(0.0, 20.0, 0.0),
                            Point3d::new(0.0, 0.0, -20.0),
                        ]),
                        &rvec,
                        &tvec,
                        &camera_matrix,
                        &dist_coeffs,
                        &mut image_points,
                    )
                    .unwrap();
                    let origin = image_points.get(0).unwrap();
                    let origin = Point2i::new(origin.x as i32, origin.y as i32);

                    for i in 0..3 {
                        let pnt = image_points.get(i + 1).unwrap();
                        let pnt = Point2i::new(pnt.x as i32, pnt.y as i32);
                    }
                }
            }

            self.stop_sig.store(true, Ordering::Relaxed);
        })
    }
}