mod cam_op;
mod cam_thread;
mod trt_thread;
mod shared_buffer;
mod trt_op;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};

fn set_ctrlc(stop_sig: Arc<AtomicBool>) {
    ctrlc::set_handler(move || {
        stop_sig.store(true, Ordering::Relaxed);
    })
    .expect("Failed to set Ctrl-C handler");
}
fn main() {
    let img_buffer = cam_thread::init_cam_thread().expect("Failed to initialize camera thread");
    let img_buffer_ref = Arc::new(img_buffer);

    let stop_sig = Arc::new(AtomicBool::new(false));
    set_ctrlc(stop_sig.clone());

    let cam_thread_handle = cam_thread::cam_thread(img_buffer_ref.clone(), stop_sig);
    cam_thread_handle.join().unwrap();

    println!("Main Thread: ending...");
    cam_thread::uninit_cam_thread();
}
