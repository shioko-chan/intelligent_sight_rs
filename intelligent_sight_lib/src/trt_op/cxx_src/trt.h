#ifndef TRT_WRAPPER_H
#define TRT_WRAPPER_H

#include <NvInfer.h>
#include <cstdint>
#include <cuda_runtime_api.h>
#include <thrust/device_ptr.h>

constexpr uint16_t MAX_DETECT = 25;
constexpr float CONF_THRESHOLD = 0.5;
constexpr float IOU_THRESHOLD = 0.5;

#define check_status(fun)                \
    do                                   \
    {                                    \
        int ret_status = (fun);          \
        if (ret_status != cudaSuccess)   \
        {                                \
            return (uint16_t)ret_status; \
        }                                \
    } while (0)

enum TrtErrCode
{
    TRT_OK = 0,
    TRT_CREATE_ENGINE_FAIL = 10000,
    TRT_CREATE_RUNTIME_FAIL,
    TRT_CREATE_CONTEXT_FAIL,
    TRT_READ_ENGINE_FILE_FAIL,
    TRT_INFER_FAIL,
    TRT_DESTROY_ENGINE_FAIL,
    TRT_CREATE_CUDASTREAM_FAIL,
    TRT_ENGINE_NOT_INITIALIZED,
    TRT_ENGINE_ALREADY_CREATED,
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
    cudaStream_t CUDA_STREAM;
    nvinfer1::ICudaEngine *M_ENGINE = nullptr;
    nvinfer1::IExecutionContext *CONTEXT = nullptr;
    nvinfer1::IRuntime *RUNTIME = nullptr;
    uint32_t WIDTH, HEIGHT;
    const char *ENGINE_NAME, *INPUT_NAME, *OUTPUT_NAME;

public:
    TensorrtInfer(const char *engine_filename, const char *input_name, const char *output_name, uint32_t width, uint32_t height);
    ~TensorrtInfer();
    uint16_t release_resources();
    uint16_t create_engine();
    uint16_t infer();
    uint16_t create_context();
    uint16_t set_input(float *input_buffer);
    uint16_t set_output(float *output_buffer);
};

struct PostProcess
{
private:
    float *transformed, *host_transformed;
    int *indices, *host_indices;
    thrust::device_ptr<int> d_indices;
    thrust::device_ptr<float> d_transformed;
    bool check_iou(float *box1, float *box2);

public:
    uint16_t init();
    uint16_t post_process(float *input_buffer, float *output_buffer, uint16_t *num_detections);
    uint16_t uninit();
};

extern "C"
{
    uint16_t create_engine(const char *engine_filename, const char *input_name, const char *output_name, uint32_t width, uint32_t height);
    uint16_t create_context();
    uint16_t infer();
    uint16_t release_resources();
    uint16_t set_input(float *input_buffer);
    uint16_t set_output(float *output_buffer);
    uint16_t postprocess_init();
    uint16_t postprocess(float *input_buffer, float *output_buffer, uint16_t *num_detections);
    uint16_t postprocess_destroy();
}
#endif