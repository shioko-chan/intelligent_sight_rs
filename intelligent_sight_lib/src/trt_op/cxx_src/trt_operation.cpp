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

uint16_t TensorrtInfer::create_engine(const char *engine_filename, uint32_t width, uint32_t height)
{
    // Deserialize engine from file
    std::ifstream engineFile(engine_filename, std::ios::binary);
    if (engineFile.fail())
    {
        return TRT_READ_ENGINE_FILE_FAIL;
    }

    engineFile.seekg(0, std::ifstream::end);

    long fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(fsize);

    engineFile.read(engineData.data(), fsize);

    engineFile.close();

    printf("TRT: Engine file size: %ld\n", fsize);

    // check_status(cudaStreamCreate(&CUDA_STREAM));
    printf("TRT: Created CUDA stream\n");
    RUNTIME = nvinfer1::createInferRuntime(G_LOGGER);
    if (RUNTIME == nullptr)
    {
        return TRT_CREATE_RUNTIME_FAIL;
    }
    printf("TRT: Created runtime\n");

    M_ENGINE = RUNTIME->deserializeCudaEngine(engineData.data(), fsize);
    if (M_ENGINE == nullptr)
    {
        return TRT_CREATE_ENGINE_FAIL;
    }
    printf("TRT: Deserialized engine\n");

    WIDTH = width;
    HEIGHT = height;

    return TRT_OK;
}

uint16_t TensorrtInfer::create_context()
{
    if (M_ENGINE)
    {
        CONTEXT = M_ENGINE->createExecutionContext();
    }
    else
    {
        return TRT_ENGINE_NOT_INITIALIZED;
    }
    if (CONTEXT == nullptr)
    {
        return TRT_CREATE_CONTEXT_FAIL;
    }
    printf("TRT: Created context\n");
    return TRT_OK;
}
uint16_t TensorrtInfer::infer(float *input_buffer, float *output_buffer)
{
    // if (!CONTEXT->setTensorAddress("images", (void *)input_buffer))
    // {
    //     G_LOGGER.log(nvinfer1::ILogger::Severity::kERROR, "Failed to set input tensor address");
    //     return TRT_INFER_FAIL;
    // }

    // if (!CONTEXT->setInputShape("images", nvinfer1::Dims4{1, 3, 640, 640}))
    // {
    //     G_LOGGER.log(nvinfer1::ILogger::Severity::kERROR, "Failed to set input shape");
    //     return TRT_INFER_FAIL;
    // }

    // if (!CONTEXT->setTensorAddress("output0", (void *)output_buffer))
    // {
    //     G_LOGGER.log(nvinfer1::ILogger::Severity::kERROR, "Failed to set output tensor address");
    //     return TRT_INFER_FAIL;
    // }

    // void *binding[2] = {input_buffer, output_buffer};
    // CONTEXT->executeV2(binding);
    // if (!CONTEXT->enqueueV3(CUDA_STREAM))
    // {
    //     G_LOGGER.log(nvinfer1::ILogger::Severity::kERROR, "Failed to enqueue");
    //     return TRT_INFER_FAIL;
    // }

    // check_status(cudaStreamSynchronize(CUDA_STREAM));

    return TRT_OK;
}

TensorrtInfer::~TensorrtInfer()
{
    // Release resources
    // cudaStreamDestroy(CUDA_STREAM);
    delete CONTEXT;
    delete M_ENGINE;
    delete RUNTIME;
}

uint16_t create_engine(const char *engine_filename, uint32_t width, uint32_t height)
{
    TRT_INFER = new TensorrtInfer();
    return TRT_INFER->create_engine(engine_filename, width, height);
}

uint16_t create_context()
{
    if (TRT_INFER != nullptr)
    {
        return TRT_INFER->create_context();
    }
    else
    {
        return TRT_ENGINE_NOT_INITIALIZED;
    }
}

uint16_t infer(float *input_buffer, float *output_buffer)
{
    return TRT_INFER->infer(input_buffer, output_buffer);
}

uint16_t release_resources()
{
    delete TRT_INFER;
    return TRT_OK;
}
