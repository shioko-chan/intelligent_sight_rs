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
    uint16_t cuda_free(uint8_t *buffer);
}

#endif