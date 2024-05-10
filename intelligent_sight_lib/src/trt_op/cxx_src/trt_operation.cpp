#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>
#include <cstdint>

#include <NvInfer.h>

#include "trt.h"

TensorrtInfer *TRT_INFER = nullptr;

void Logger::log(Severity severity, const char *msg) noexcept
{
    if (severity <= Severity::kVERBOSE)
    {
        std::cout << static_cast<int32_t>(severity) << ": " << msg << std::endl;
    }
}

uint8_t TensorrtInfer::create_engine(const char *engine_filename, uint32_t width, uint32_t height)
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

    if (cudaStreamCreate(CUDA_STREAM) != cudaSuccess)
    {
        return TRT_CREATE_CUDASTREAM_FAIL;
    }

    RUNTIME = nvinfer1::createInferRuntime(G_LOGGER);
    if (RUNTIME == nullptr)
    {
        return TRT_CREATE_RUNTIME_FAIL;
    }

    M_ENGINE = RUNTIME->deserializeCudaEngine(engineData.data(), fsize);
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

uint8_t TensorrtInfer::infer(float *input_buffer, float *output_buffer)
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

    if (!CONTEXT->enqueueV3(*CUDA_STREAM))
    {
        G_LOGGER.log(nvinfer1::ILogger::Severity::kERROR, "Failed to enqueue");
    }

    return TRT_OK;
}

TensorrtInfer::~TensorrtInfer()
{
    // Release resources
    cudaStreamDestroy(*CUDA_STREAM);
    delete CONTEXT;
    delete M_ENGINE;
    delete RUNTIME;
}

uint8_t create_engine(const char *engine_filename, uint32_t width, uint32_t height)
{
    TRT_INFER = new TensorrtInfer();
    return TRT_INFER->create_engine(engine_filename, width, height);
}

uint8_t infer(float *input_buffer, float *output_buffer)
{
    return TRT_INFER->infer(input_buffer, output_buffer);
}

uint8_t destroy_engine()
{
    delete TRT_INFER;
    return TRT_OK;
}
