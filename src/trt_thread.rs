use crate::thread_trait::Processor;
use anyhow::Result;
use intelligent_sight_lib::{create_engine, infer, Image, SharedBuffer, Tensor};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
pub struct TrtThread {
    input_buffer: Arc<SharedBuffer<Image>>,
    output_buffer: Arc<SharedBuffer<Tensor>>,
    stop_sig: Arc<AtomicBool>,
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
            let mut cnt = 0;
            let mut start = std::time::Instant::now();
            let mut engine_input_buffer = Tensor::new(1 * 3 * 640 * 640).unwrap();
            while stop_sig.load(Ordering::Relaxed) == false {
                let lock_input = input_buffer.read();
                let mut lock_output = output_buffer.write();
                match infer(&engine_input_buffer.data, &mut lock_output.data) {
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
                output_buffer.write_finish(lock_output);
                input_buffer.read_finish(lock_input);
            }
        })
    }
}

impl TrtThread {
    pub fn new(input_buffer: Arc<SharedBuffer<Image>>, stop_sig: Arc<AtomicBool>) -> Result<Self> {
        create_engine("model.trt", 640, 640)?;
        Ok(TrtThread {
            input_buffer,
            output_buffer: Arc::new(SharedBuffer::new_with_default(
                4,
                Tensor::new(1 * 31 * 8400)?,
            )),
            stop_sig,
        })
    }
}
