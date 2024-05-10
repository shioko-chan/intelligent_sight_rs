#ifndef TRT_WRAPPER_H
#define TRT_WRAPPER_H

#include <NvInfer.h>
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

enum TrtErrCode
{
    TRT_OK = 0,
    TRT_CREATE_ENGINE_FAIL,
    TRT_CREATE_RUNTIME_FAIL,
    TRT_CREATE_CONTEXT_FAIL,
    TRT_READ_ENGINE_FILE_FAIL,
    TRT_INFER_FAIL,
    TRT_DESTROY_ENGINE_FAIL,
    TRT_CREATE_CUDASTREAM_FAIL,
};

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override;
};

struct TensorrtInfer
{
private:
    Logger G_LOGGER;
    cudaStream_t *CUDA_STREAM = nullptr;
    nvinfer1::ICudaEngine *M_ENGINE = nullptr;
    nvinfer1::IExecutionContext *CONTEXT = nullptr;
    nvinfer1::IRuntime *RUNTIME = nullptr;
    uint32_t WIDTH, HEIGHT;

public:
    ~TensorrtInfer();
    uint8_t create_engine(const char *engine_filename, uint32_t width, uint32_t height);
    uint8_t infer(float *input_buffer, float *output_buffer);
};

extern "C"
{
    uint8_t create_engine(const char *engine_filename, uint32_t width, uint32_t height);
    uint8_t infer(float *input_buffer, float *output_buffer);
    uint8_t destroy_engine();
}
#endif