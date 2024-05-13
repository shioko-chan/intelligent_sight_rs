#include <cuda_runtime.h>
#include <cfloat>
#include <cstdint>
#include "cuda_op.h"

cudaStream_t CUDASTREAM;

__global__ void rgbToTensor(unsigned char *input, float *output, uint32_t width, uint32_t height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height + 160)
    {
        int idx_out = (y * width + x);
        int size_out = width * (height + 160);

        if (y < height + 80 && y >= 80)
        {
            int idx_in = 3 * ((y - 80) * width + x);

            output[idx_out] = input[idx_in] / 255.0f;                    // R
            output[idx_out + size_out] = input[idx_in + 1] / 255.0f;     // G
            output[idx_out + 2 * size_out] = input[idx_in + 2] / 255.0f; // B
        }
        else
        {
            output[idx_out] = 0.5f;                // R
            output[idx_out + size_out] = 0.5f;     // G
            output[idx_out + 2 * size_out] = 0.5f; // B
        }
    }
}

// assume that output is (640, 640), input is (640, 480), padding 80 pixels on top and bottom
uint16_t convert_rgb888_3dtensor(uint8_t *input_buffer, float *output_buffer, uint32_t width, uint32_t height)
{
    dim3 threads_per_block(16, 16);
    dim3 num_blocks(40, 40);
    rgbToTensor<<<num_blocks, threads_per_block>>>(input_buffer, output_buffer, width, height);
    cudaDeviceSynchronize();
    return (uint16_t)cudaSuccess;
}

uint16_t transfer_host_to_device(uint8_t *host_mem, uint8_t *device_mem, uint32_t size)
{
    check_status(cudaMemcpy(device_mem, host_mem, size, cudaMemcpyHostToDevice));
    return (uint16_t)cudaSuccess;
}

uint16_t transfer_device_to_host(uint8_t *host_mem, uint8_t *device_mem, uint32_t size)
{
    check_status(cudaMemcpy(host_mem, device_mem, size, cudaMemcpyDeviceToHost));
    return (uint16_t)cudaSuccess;
}

uint16_t cuda_malloc(uint32_t size, uint8_t **buffer)
{
    check_status(cudaMalloc((void **)buffer, size));
    return (uint16_t)cudaSuccess;
}

uint16_t cuda_malloc_managed(uint32_t size, uint8_t **buffer)
{
    check_status(cudaMallocManaged((void **)buffer, size));
    return (uint16_t)cudaSuccess;
}

uint16_t cuda_free(uint8_t *buffer)
{
    check_status(cudaFree(buffer));
    return (uint16_t)cudaSuccess;
}

uint16_t init_cuda()
{
    check_status(cudaStreamCreate(&CUDASTREAM));
    return (uint16_t)cudaSuccess;
}

uint16_t destroy_cuda()
{
    check_status(cudaStreamDestroy(CUDASTREAM));
    return (uint16_t)cudaSuccess;
}