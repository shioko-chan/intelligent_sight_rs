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

uint16_t TensorrtInfer::create_engine()
{
    // Deserialize engine from file
    std::ifstream engineFile(ENGINE_NAME, std::ios::binary);
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

    check_status(cudaStreamCreate(&CUDA_STREAM));

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

uint16_t TensorrtInfer::set_input(float *input_buffer)
{
    if (!CONTEXT->setTensorAddress(INPUT_NAME, (void *)input_buffer))
    {
        G_LOGGER.log(nvinfer1::ILogger::Severity::kERROR, "Failed to set input tensor address");
        return TRT_INFER_FAIL;
    }
}

uint16_t TensorrtInfer::set_output(float *output_buffer)
{
    if (!CONTEXT->setTensorAddress(OUTPUT_NAME, (void *)output_buffer))
    {
        G_LOGGER.log(nvinfer1::ILogger::Severity::kERROR, "Failed to set output tensor address");
        return TRT_INFER_FAIL;
    }
}

uint16_t TensorrtInfer::infer()
{
    // void *binding[2] = {input_buffer, output_buffer};
    // CONTEXT->executeV2(binding);
    if (!CONTEXT->enqueueV3(CUDA_STREAM))
    {
        G_LOGGER.log(nvinfer1::ILogger::Severity::kERROR, "Failed to enqueue");
        return TRT_INFER_FAIL;
    }

    check_status(cudaStreamSynchronize(CUDA_STREAM));

    return TRT_OK;
}

TensorrtInfer::TensorrtInfer(const char *engine_filename, const char *input_name, const char *output_name, uint32_t width, uint32_t height) : ENGINE_NAME(engine_filename), INPUT_NAME(input_name), OUTPUT_NAME(output_name), WIDTH(width), HEIGHT(height) {}

TensorrtInfer::~TensorrtInfer()
{
    // Release resources
    cudaStreamDestroy(CUDA_STREAM);
    delete CONTEXT;
    delete M_ENGINE;
    delete RUNTIME;
}

uint16_t create_engine(const char *engine_filename, const char *input_name, const char *output_name, uint32_t width, uint32_t height)
{
    TRT_INFER = new TensorrtInfer(engine_filename, input_name, output_name, width, height);
    return TRT_INFER->create_engine();
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

uint16_t set_input(float *input_buffer)
{
    return TRT_INFER->set_input(input_buffer);
}
uint16_t set_output(float *output_buffer)
{
    return TRT_INFER->set_output(output_buffer);
}

uint16_t infer()
{
    return TRT_INFER->infer();
}

uint16_t release_resources()
{
    delete TRT_INFER;
    return TRT_OK;
}
