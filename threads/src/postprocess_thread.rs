use crate::thread_trait::Processor;
use anyhow::Result;
use intelligent_sight_lib::{
    postprocess, postprocess_destroy, postprocess_init, Reader, TensorBuffer, Writer,
};
use log::{debug, error, info, log_enabled};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

#[cfg(feature = "visualize")]
use std::sync::mpsc;

pub struct PostprocessThread {
    input_buffer: Reader<TensorBuffer>,
    output_buffer: Writer<TensorBuffer>,
    stop_sig: Arc<AtomicBool>,
    #[cfg(feature = "visualize")]
    detection_tx: std::sync::mpsc::Sender<TensorBuffer>,
}

impl Drop for PostprocessThread {
    fn drop(&mut self) {
        if let Err(err) = postprocess_destroy() {
            error!("PostprocessThread: Failed to release resources: {}", err);
        }
    }
}

impl Processor for PostprocessThread {
    type Output = TensorBuffer;

    fn get_output_buffer(&self) -> Reader<Self::Output> {
        self.output_buffer.get_reader()
    }

    fn start_processor(self) -> thread::JoinHandle<()> {
        thread::spawn(move || {
            let mut cnt = 0;
            let mut start = std::time::Instant::now();

            while self.stop_sig.load(Ordering::Relaxed) == false {
                let mut lock_input = match self.input_buffer.read() {
                    Some(input) => input,
                    None => {
                        if self.stop_sig.load(Ordering::Relaxed) == false {
                            error!("PostprocessThread: Failed to get input");
                        }
                        break;
                    }
                };

                let mut lock_output = self.output_buffer.write();
                lock_output.timestamp = lock_input.timestamp;

                match postprocess(&mut lock_input, &mut lock_output) {
                    #[cfg(feature = "visualize")]
                    Ok(cnt) => {
                        let mut det = lock_output.clone();
                        det.resize(vec![cnt as usize, 16]);
                        if let Err(err) = self.detection_tx.send(det) {
                            if self.stop_sig.load(Ordering::Relaxed) == false {
                                error!("PostprocessThread: Failed to send detection: {}", err);
                            }
                            break;
                        }
                    }
                    #[cfg(not(feature = "visualize"))]
                    Ok(_) => {}
                    Err(err) => {
                        error!("PostprocessThread: Failed to postprocess: {}", err);
                        break;
                    }
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
            self.stop_sig.store(true, Ordering::Relaxed);
        })
    }
}

impl PostprocessThread {
    pub fn new(
        input_buffer: Reader<TensorBuffer>,
        stop_sig: Arc<AtomicBool>,
        #[cfg(feature = "visualize")] detection_tx: mpsc::Sender<TensorBuffer>,
    ) -> Result<Self> {
        postprocess_init()?;

        info!("PostprocessThread: output buffer size: {:?}", vec![25, 16]);

        Ok(Self {
            input_buffer,
            output_buffer: Writer::new(4, || TensorBuffer::new(vec![25, 16]))?,
            stop_sig,
            #[cfg(feature = "visualize")]
            detection_tx,
        })
    }
}
