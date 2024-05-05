use crate::cam_op::{get_image, initialize_camera, uninitialize_camera, FlipFlag, Image};
use crate::shared_buffer::SharedBuffer;
use anyhow::{anyhow, Result};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

pub fn init_cam_thread() -> Result<SharedBuffer<Image>> {
    let mut buffer_width = vec![0u32; 1];
    let mut buffer_height = vec![0u32; 1];
    let mut initialize_retry = 0;

    while let Err(err) = initialize_camera(1, &mut buffer_width, &mut buffer_height) {
        println!("Failed to initialize camera with err: {}, retrying...", err);
        initialize_retry += 1;
        if initialize_retry > 10 {
            println!("Failed to initialize camera after 10 retries, exiting...");
            return Err(anyhow!("Failed to initialize camera"));
        }
    }
    Ok(SharedBuffer::<Image>::new_with_default(
        4,
        Image::new(buffer_width[0], buffer_height[0]),
    ))
}

pub fn cam_thread(
    shared_buffer: Arc<SharedBuffer<Image>>,
    stop_sig: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut cnt = 0;
        let mut start = std::time::Instant::now();
        while stop_sig.load(Ordering::Relaxed) == false {
            let mut lock = shared_buffer.write();
            match get_image(0, &mut lock, FlipFlag::None) {
                Ok(_) => {}
                Err(err) => {
                    println!("err: {}", err);
                }
            }
            cnt += 1;
            if cnt == 10 {
                let end = std::time::Instant::now();
                let elapsed = end.duration_since(start);
                println!("fps: {}", 10.0 / elapsed.as_secs_f32());
                start = end;
                cnt = 0;
            }
            // println!("{} {}", lock.width, lock.height);
            shared_buffer.write_finish(lock);
        }
    })
}

pub fn uninit_cam_thread() {
    let mut uninitialize_retry = 0;
    while let Err(err) = uninitialize_camera() {
        println!(
            "Failed to uninitialize camera with err: {}, retrying...",
            err
        );
        uninitialize_retry += 1;
        if uninitialize_retry > 10 {
            println!("Failed to uninitialize camera after 10 retries, exiting...");
            return;
        }
    }
}
