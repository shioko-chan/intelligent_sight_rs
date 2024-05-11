#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>
#include "cuda_op.h"

__global__ void rgbToTensor(unsigned char *input, float *output, uint32_t width, uint32_t height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        // Compute the index for the 3D arrays
        int idx_in = 3 * (y * width + x);
        int idx_out = (y * width + x);

        // Convert RGB to tensor
        output[idx_out] = input[idx_in] / 255.0f;                          // R
        output[idx_out + width * height] = input[idx_in + 1] / 255.0f;     // G
        output[idx_out + 2 * width * height] = input[idx_in + 2] / 255.0f; // B
    }
}

uint16_t convert_rgb888_3dtensor(uint8_t *input_buffer, float *output_buffer, uint32_t width, uint32_t height)
{
    dim3 threads_per_block(16, 16);
    dim3 num_blocks((width + threads_per_block.x - 1) / threads_per_block.x, (height + threads_per_block.y - 1) / threads_per_block.y);
    rgbToTensor<<<num_blocks, threads_per_block>>>(input_buffer, output_buffer, width, height);
    return CUDA_OK;
}

uint16_t cuda_malloc(uint32_t size, uint8_t **buffer)
{
    check_status(cudaMallocManaged((void **)buffer, size));
    return (uint16_t)cudaSuccess;
}

uint16_t cuda_free(uint8_t *buffer)
{
    check_status(cudaFree(buffer));
    return (uint16_t)cudaSuccess;
}