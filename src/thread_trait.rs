use anyhow::Result;
use std::thread::JoinHandle;

pub trait Processor
where
    Self: Sized,
{
    type Output;
    fn get_output_buffer(
        &self,
    ) -> std::sync::Arc<intelligent_sight_lib::SharedBuffer<Self::Output>>;
    fn start_processor(&self) -> JoinHandle<()>;
    fn clean_up(self) -> Result<()> {
        Ok(())
    }
}
