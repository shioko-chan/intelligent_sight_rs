#ifndef CUDA_WRAPPER_H
#define CUDA_WRAPPER_H

#include <cstdint>
#include <cuda_runtime_api.h>

#define check_status(fun)                \
    do                                   \
    {                                    \
        int ret_status = (fun);          \
        if (ret_status != cudaSuccess)   \
        {                                \
            return (uint16_t)ret_status; \
        }                                \
    } while (0)

extern "C"
{
    uint16_t cuda_malloc(uint32_t size, uint8_t **buffer);
    uint16_t cuda_malloc_managed(uint32_t size, uint8_t **buffer);
    uint16_t cuda_malloc_host(uint32_t size, uint8_t **buffer);
    uint16_t cuda_free(uint8_t *buffer);
    uint16_t cuda_free_host(uint8_t *buffer);
    uint16_t convert_rgb888_3dtensor(uint8_t *input_buffer, float *output_buffer, uint32_t width, uint32_t height);
    uint16_t init_cuda();
    uint16_t destroy_cuda();
    uint16_t transfer_host_to_device(uint8_t *host_mem, uint8_t *device_mem, uint32_t size);
    uint16_t transfer_device_to_host(uint8_t *host_mem, uint8_t *device_mem, uint32_t size);
}

#endif