mod cam_op;
mod cuda_op;
mod data_structures;
mod shared_buffer;
mod trt_op;
mod unified_item;

pub use cam_op::FlipFlag;
pub use cam_op::{get_image, initialize_camera, uninitialize_camera};

#[cfg(target_os = "linux")]
pub use cuda_op::cuda_malloc_managed;

pub use cuda_op::{
    convert_rgb888_3dtensor, cuda_free, transfer_device_to_host, transfer_host_to_device,
};
#[cfg(target_os = "windows")]
pub use cuda_op::{cuda_free_host, cuda_malloc, cuda_malloc_host};
pub use data_structures::{Image, Tensor};
pub use shared_buffer::SharedBuffer;
pub use trt_op::{create_context, create_engine, infer, release_resources, set_input, set_output};
pub use unified_item::{UnifiedItem, UnifiedTrait};
