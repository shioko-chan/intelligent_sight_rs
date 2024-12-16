use crate::thread_trait::Processor;
use anyhow::{anyhow, Result};
use intelligent_sight_lib::{Detection, DetectionBuffer, Reader, Writer};
use log::error;
use opencv::{self as cv, core::*};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    thread::{self, JoinHandle},
};

macro_rules! generate_points {
    ($width:expr, $height:expr) => {
        [
            Point3d::new($width / 2.0, -$height / 2.0, 0.0),
            Point3d::new($width / 2.0, $height / 2.0, 0.0),
            Point3d::new(-$width / 2.0, $height / 2.0, 0.0),
            Point3d::new(-$width / 2.0, -$height / 2.0, 0.0),
        ]
    };
}

pub struct AnalysisThread {
    input_buffer: Reader<DetectionBuffer>,
    output_buffer: Writer<Vec<f32>>,
    stop_sig: Arc<AtomicBool>,
}

impl AnalysisThread {
    #[allow(unused)]
    const CLASSES: [&'static str; 18] = [
        "PR", "B1", "B2", "B3", "B4", "B5", "BG", "BO", "BB", "R1", "R2", "R3", "R4", "R5", "RG",
        "RO", "RB", "PB",
    ];

    const POWER_RUNE_WIDTH: f64 = 320.0;
    const POWER_RUNE_HEIGHT: f64 = 102.6;
    const POWER_RUNE_POINTS: [Point3_<f64>; 4] =
        generate_points!(Self::POWER_RUNE_WIDTH, Self::POWER_RUNE_HEIGHT);

    const ARMOR_WIDTH: f64 = 135.0; // 1.08
    const ARMOR_HEIGHT: f64 = 125.0;
    const ARMOR_POINTS: [Point3_<f64>; 4] = generate_points!(Self::ARMOR_WIDTH, Self::ARMOR_HEIGHT);

    const LARGE_ARMOR_WIDTH: f64 = 230.0; // 1.81
    const LARGE_ARMOR_HEIGHT: f64 = 127.0;
    const LARGE_ARMOR_POINTS: [Point3_<f64>; 4] =
        generate_points!(Self::LARGE_ARMOR_WIDTH, Self::LARGE_ARMOR_HEIGHT);

    #[allow(unused)]
    const COLORS: [VecN<f64, 4>; 5] = [
        VecN::new(0.0, 0.0, 255.0, 255.0),
        VecN::new(0.0, 255.0, 0.0, 255.0),
        VecN::new(255.0, 0.0, 0.0, 255.0),
        VecN::new(255.0, 255.0, 0.0, 255.0),
        VecN::new(255.0, 0.0, 255.0, 255.0),
    ];

    pub fn new(input_buffer: Reader<DetectionBuffer>, stop_sig: Arc<AtomicBool>) -> Result<Self> {
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

    fn start_processor(self) -> JoinHandle<()> {
        thread::spawn(move || {
            // 准备3D点（物体坐标系）
            let power_rune_points = Vector::from_slice(&Self::POWER_RUNE_POINTS);
            let armor_points = Vector::from_slice(&Self::ARMOR_POINTS);
            let large_armor_points = Vector::from_slice(&Self::LARGE_ARMOR_POINTS);

            // 定义相机内参矩阵
            let camera_matrix =
                Mat::from_slice_2d(&[[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]])
                    .unwrap();

            // 畸变系数（假设无畸变）
            let dist_coeffs = Mat::zeros(4, 1, cv::core::CV_64F).unwrap();

            while self.stop_sig.load(Ordering::Relaxed) == false {
                let Some(lock_input) = self.input_buffer.read() else {
                    if self.stop_sig.load(Ordering::Relaxed) == false {
                        error!("AnalysisThread: Failed to get input");
                    }
                    break;
                };
                for Detection {
                    conf,
                    w,
                    h,
                    cls,
                    points,
                    ..
                } in lock_input.iter()
                {
                    let mut image_points = Vector::<Point2d>::with_capacity(4);

                    let object_points = match *cls {
                        0 | 17 => {
                            for (i, [x, y]) in points.iter().enumerate() {
                                if i != 2 {
                                    image_points.push(Point2d::new(*x as f64, *y as f64));
                                }
                            }
                            &power_rune_points
                        }
                        c @ _ => {
                            for (i, [x, y]) in points.iter().enumerate() {
                                if i != 4 {
                                    image_points.push(Point2d::new(*x as f64, *y as f64));
                                }
                            }
                            if c == 1 || c == 9 {
                                &large_armor_points
                            } else if w / h > 1.5 {
                                &large_armor_points
                            } else {
                                &armor_points
                            }
                        }
                    };

                    // 旋转向量和平移向量
                    let mut rvec = Mat::default();
                    let mut tvec = Mat::default();
                    cv::calib3d::solve_pnp(
                        object_points,
                        &image_points,
                        &camera_matrix,
                        &dist_coeffs,
                        &mut rvec,
                        &mut tvec,
                        false,
                        cv::calib3d::SOLVEPNP_IPPE,
                    )
                    .unwrap();
                }
            }

            self.stop_sig.store(true, Ordering::Relaxed);
        })
    }
}
