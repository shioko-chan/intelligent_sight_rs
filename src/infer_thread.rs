use crate::thread_trait::Processor;
use anyhow::Result;
use intelligent_sight_lib::{
    convert_rgb888_3dtensor, create_context, create_engine, infer, release_resources, set_input,
    set_output, Image, SharedBuffer, Tensor,
};
use log::{debug, info, log_enabled, warn};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

pub struct TrtThread {
    input_buffer: Arc<SharedBuffer<Image>>,
    output_buffer: Arc<SharedBuffer<Tensor>>,
    stop_sig: Arc<AtomicBool>,
}

impl Drop for TrtThread {
    fn drop(&mut self) {
        if let Err(err) = release_resources() {
            warn!("InferThread: Failed to release resources: {}", err);
        }
    }
}

impl Processor for TrtThread {
    type Output = Tensor;

    fn get_output_buffer(
        &self,
    ) -> std::sync::Arc<intelligent_sight_lib::SharedBuffer<Self::Output>> {
        self.output_buffer.clone()
    }

    fn start_processor(&self) -> thread::JoinHandle<()> {
        let input_buffer = self.input_buffer.clone();
        let output_buffer = self.output_buffer.clone();
        let stop_sig = self.stop_sig.clone();

        thread::spawn(move || {
            if let Err(err) = create_context() {
                warn!(
                    "InferThread: failed create engine, context, runtime due to error: {}",
                    err
                );
                return;
            }
            let mut cnt = 0;
            let mut start = std::time::Instant::now();
            let mut engine_input_buffer = Tensor::new(vec![1, 3, 640, 640]).unwrap();

            info!(
                "InferThread: middle buffer size: {:?}",
                engine_input_buffer.size(),
            );

            while stop_sig.load(Ordering::Relaxed) == false {
                let mut lock_input = input_buffer.read();
                if let Err(err) = convert_rgb888_3dtensor(&mut lock_input, &mut engine_input_buffer)
                {
                    warn!("InferThread: convert image to tensor failed {}", err);
                }
                input_buffer.read_finish(lock_input);

                if log_enabled!(log::Level::Trace) {
                    trace!("InferThread: finish convert_rgb888_3dtensor");
                }

                if let Err(err) = set_input(&mut engine_input_buffer) {
                    warn!("InferThread: set input buffer failed, error {}", err);
                }

                let mut lock_output = output_buffer.write();
                if let Err(err) = set_output(&mut lock_output) {
                    warn!("InferThread: set output buffer failed, error {}", err);
                }
                if let Err(err) = infer() {
                    warn!("InferThread: infer failed, error {}", err);
                }
                output_buffer.write_finish(lock_output);

                if log_enabled!(log::Level::Debug) {
                    cnt += 1;
                    if cnt == 10 {
                        let end = std::time::Instant::now();
                        let elapsed = end.duration_since(start);
                        debug!("InferThread: fps: {}", 10.0 / elapsed.as_secs_f32());
                        start = end;
                        cnt = 0;
                    }
                }
            }
        })
    }
}

impl TrtThread {
    pub fn new(input_buffer: Arc<SharedBuffer<Image>>, stop_sig: Arc<AtomicBool>) -> Result<Self> {
        create_engine("model.trt", "images", "output0", 640, 640)?;
        let read_lock = input_buffer.read();
        info!(
            "InferThread: input buffer size: width: {}, height: {}",
            read_lock.width, read_lock.height
        );
        input_buffer.read_finish(read_lock);
        info!("InferThread: output buffer size: {:?}", vec![1, 31, 8400]);
        Ok(TrtThread {
            input_buffer,
            output_buffer: Arc::new(SharedBuffer::new(4, || Tensor::new(vec![1, 31, 8400]))?),
            stop_sig,
        })
    }
}
