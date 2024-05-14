mod cam_thread;
mod infer_thread;
mod thread_trait;
use env_logger::{Builder, Target};
use log::{info, warn};
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
    Builder::from_default_env().target(Target::Stdout).init();

    info!("Main: starting...🚀");

    let stop_sig = Arc::new(AtomicBool::new(false));

    let camera_thread = match cam_thread::CamThread::new(stop_sig.clone()) {
        Ok(camera_thread) => camera_thread,
        Err(err) => {
            warn!("Main: Failed to initialize camera thread: {}", err);
            return;
        }
    };
    info!("Main: Camera thread initialized");

    let infer_thread =
        match infer_thread::TrtThread::new(camera_thread.get_output_buffer(), stop_sig.clone()) {
            Ok(infer_thread) => infer_thread,
            Err(err) => {
                warn!("Main: Failed to initialize infer thread: {}", err);
                return;
            }
        };
    info!("Main: Infer thread initialized");

    set_ctrlc(stop_sig.clone());

    let camera_thread_handle = camera_thread.start_processor();
    info!("Main: Camera thread started");

    let infer_thread_handle = infer_thread.start_processor();
    info!("Main: Infer thread started");

    camera_thread_handle.join().unwrap();
    infer_thread_handle.join().unwrap();

    info!("Main: ending...");
}
