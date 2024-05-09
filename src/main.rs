mod cam_thread;
mod thread_trait;
mod trt_thread;

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use thread_trait::Processor;

fn set_ctrlc(stop_sig: Arc<AtomicBool>) {
    ctrlc::set_handler(move || {
        stop_sig.store(true, Ordering::Relaxed);
    })
    .expect("Failed to set Ctrl-C handler");
}

fn main() {
    let stop_sig = Arc::new(AtomicBool::new(false));
    let Ok(camera_thread) = cam_thread::CamThread::new(stop_sig.clone()) else {
        println!("Failed to initialize camera thread");
        return;
    };
    let Ok(infer_thread) =
        trt_thread::TrtThread::new(camera_thread.get_output_buffer(), stop_sig.clone())
    else {
        println!("Failed to initialize trt thread");
        return;
    };
    set_ctrlc(stop_sig.clone());
    let camera_thread_handle = camera_thread.start_processor();
    let infer_thread_handle = infer_thread.start_processor();

    camera_thread_handle.join().unwrap();
    infer_thread_handle.join().unwrap();

    println!("Main Thread: ending...");

    camera_thread.clean_up().unwrap();
    infer_thread.clean_up().unwrap();
}
