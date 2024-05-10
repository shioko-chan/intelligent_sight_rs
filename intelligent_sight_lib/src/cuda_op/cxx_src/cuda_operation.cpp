#include <cuda_runtime_api.h>
#include <stdint.h>
#include "cuda_op.h"

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