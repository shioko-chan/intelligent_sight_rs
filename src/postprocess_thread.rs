use crate::thread_trait::Processor;
use anyhow::Result;
use intelligent_sight_lib::{
    postprocess, postprocess_destroy, postprocess_init, SharedBuffer, Tensor,
};
use log::{debug, info, log_enabled, warn};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

pub struct PostprocessThread {
    input_buffer: Arc<SharedBuffer<Tensor>>,
    output_buffer: Arc<SharedBuffer<Tensor>>,
    stop_sig: Arc<AtomicBool>,
}

impl Drop for PostprocessThread {
    fn drop(&mut self) {
        if let Err(err) = postprocess_destroy() {
            warn!("PostprocessThread: Failed to release resources: {}", err);
        }
    }
}

impl Processor for PostprocessThread {
    type Output = Tensor;

    fn get_output_buffer(&self) -> Arc<SharedBuffer<Self::Output>> {
        self.output_buffer.clone()
    }

    fn start_processor(&self) -> thread::JoinHandle<()> {
        let input_buffer = self.input_buffer.clone();
        let output_buffer = self.output_buffer.clone();
        let stop_sig = self.stop_sig.clone();
        let mut cnt = 0;
        let mut start = std::time::Instant::now();

        thread::spawn(move || {
            while stop_sig.load(Ordering::Relaxed) == false {
                let mut lock_input = input_buffer.read();
                let mut lock_output = output_buffer.write();
                if let Err(err) = postprocess(&mut lock_input, &mut lock_output) {
                    warn!("PostprocessThread: Failed to postprocess: {}", err);
                    break;
                }
                drop(lock_input);
                drop(lock_output);
                if log_enabled!(log::Level::Debug) {
                    cnt += 1;
                    if cnt == 10 {
                        let end = std::time::Instant::now();
                        let elapsed = end.duration_since(start);
                        debug!("PostprocessThread: fps: {}", 10.0 / elapsed.as_secs_f32());
                        start = end;
                        cnt = 0;
                    }
                }
            }
            stop_sig.store(true, Ordering::Relaxed);
        })
    }
}

impl PostprocessThread {
    pub fn new(input_buffer: Arc<SharedBuffer<Tensor>>, stop_sig: Arc<AtomicBool>) -> Result<Self> {
        postprocess_init()?;
        let read_lock = input_buffer.read();
        info!(
            "PostprocessThread: input buffer size: {:?}",
            read_lock.size()
        );
        info!("PostprocessThread: output buffer size: {:?}", vec![25, 16]);

        drop(read_lock);
        Ok(Self {
            input_buffer,
            output_buffer: Arc::new(SharedBuffer::new(2, || Tensor::new(vec![25, 16]))?),
            stop_sig,
        })
    }
}
