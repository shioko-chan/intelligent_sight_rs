#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <cstdint>

#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include "trt.h"

class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity <= Severity::kVERBOSE)
        {
            std::cout << static_cast<int32_t>(severity) << ": " << msg << std::endl;
        }
    }
};

Logger G_LOGGER;
cudaStream_t CUDA_STREAM;
nvinfer1::ICudaEngine *M_ENGINE = nullptr;
nvinfer1::IExecutionContext *CONTEXT = nullptr;
uint32_t WIDTH, HEIGHT;

uint8_t cuda_malloc(uint32_t size, uint8_t **buffer)
{
    check_status(cudaMallocManaged((void **)buffer, size));
    return TRT_OK;
}

uint8_t cuda_free(uint8_t *buffer)
{
    check_status(cudaFree(buffer));
    return TRT_OK;
}

uint8_t create_engine(const char *engine_filename, uint32_t width, uint32_t height)
{
    // Deserialize engine from file
    std::ifstream engineFile(engine_filename, std::ios::binary);
    if (engineFile.fail())
    {
        return TRT_READ_ENGINE_FILE_FAIL;
    }

    engineFile.seekg(0, std::ifstream::end);

    auto fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(fsize);

    engineFile.read(engineData.data(), fsize);

    engineFile.close();

    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(G_LOGGER);
    if (runtime == nullptr)
    {
        return TRT_CREATE_RUNTIME_FAIL;
    }

    M_ENGINE = runtime->deserializeCudaEngine(engineData.data(), fsize);
    if (M_ENGINE == nullptr)
    {
        return TRT_CREATE_ENGINE_FAIL;
    }

    CONTEXT = M_ENGINE->createExecutionContext();
    if (CONTEXT == nullptr)
    {
        return TRT_CREATE_CONTEXT_FAIL;
    }

    WIDTH = width;
    HEIGHT = height;

    return TRT_OK;
}

uint8_t infer(float *input_buffer, float *output_buffer)
{
    // Read input image
    // for (int c = 0; c < 3; c++)
    // {
    //     for (int j = 0, HW = HEIGHT * WIDTH; j < HW; ++j)
    //     {
    //         input_buffer[c * HW + j] = (static_cast<float>(mPPM.buffer[j * 3 + c]) / mPPM.max - mMean[c]) / mStd[c];
    //     }
    // }
    if (!CONTEXT->setTensorAddress("input", input_buffer))
    {
        G_LOGGER.log(nvinfer1::ILogger::Severity::kERROR, "Failed to set input tensor address");
        return false;
    }

    if (!CONTEXT->setTensorAddress("output", output_buffer))
    {
        G_LOGGER.log(nvinfer1::ILogger::Severity::kERROR, "Failed to set output tensor address");
    }

    if (!CONTEXT->enqueueV3(CUDA_STREAM))
    {
        G_LOGGER.log(nvinfer1::ILogger::Severity::kERROR, "Failed to enqueue");
    }

    return TRT_OK;
}

uint8_t destroy_engine()
{
    // Release resources
    cudaStreamDestroy(CUDA_STREAM);
    return TRT_OK;
}