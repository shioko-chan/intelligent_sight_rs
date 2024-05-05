#include <cstdint>
#include <cuda_runtime_api.h>

#define check_status(fun)               \
    do                                  \
    {                                   \
        int ret_status = (fun);         \
        if (ret_status != cudaSuccess)  \
        {                               \
            return (uint8_t)ret_status; \
        }                               \
    } while (0)

extern "C"
{
    uint8_t cuda_malloc(uint32_t size, uint8_t **buffer);
    uint8_t cuda_free(uint8_t *buffer);
    uint8_t create_engine(const char *engine_filename, uint32_t width, uint32_t height);
}
