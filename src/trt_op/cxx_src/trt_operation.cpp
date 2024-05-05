#include <cassert>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <cuda_runtime_api.h>
#include <NvInfer.h>

#include "trt.h"

std::ostream &operator<<(std::ostream &os, const nvinfer1::ILogger::Severity &severity)
{
    switch (severity)
    {
    case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
        os << "INTERNAL_ERROR";
        break;
    case nvinfer1::ILogger::Severity::kERROR:
        os << "ERROR";
        break;
    case nvinfer1::ILogger::Severity::kWARNING:
        os << "WARNING";
        break;
    case nvinfer1::ILogger::Severity::kINFO:
        os << "INFO";
        break;
    case nvinfer1::ILogger::Severity::kVERBOSE:
        os << "VERBOSE";
        break;
    default:
        os << "UNKNOWN_SEVERITY";
        break;
    }
    return os;
}

class SimpleLogger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        // print severity and message to console
        std::cout << severity << ": " << msg << std::endl;
    }
};

SimpleLogger gLogger;

nvinfer1::ICudaEngine *M_ENGINE = nullptr;
nvinfer1::IExecutionContext *CONTEXT = nullptr;
uint32_t WIDTH, HEIGHT;

uint8_t cuda_malloc(uint32_t size, uint8_t **buffer)
{
    check_status(cudaMallocManaged((void **)buffer, size));
    return 0;
}

uint8_t cuda_free(uint8_t *buffer)
{
    check_status(cudaFree(buffer));
    return 0;
}

uint8_t create_engine(const char *engine_filename, uint32_t width, uint32_t height)
{
    // De-serialize engine from file
    std::ifstream engineFile(engine_filename, std::ios::binary);
    if (engineFile.fail())
    {
        return 1;
    }

    engineFile.seekg(0, std::ifstream::end);
    auto fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);

    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);

    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);
    M_ENGINE = runtime->deserializeCudaEngine(engineData.data(), fsize);

    if (M_ENGINE == nullptr)
    {
        return 1;
    }

    // Create execution context
    CONTEXT = M_ENGINE->createExecutionContext();
    if (CONTEXT == nullptr)
    {
        return 1;
    }
    WIDTH = width;
    HEIGHT = height;

    return 0;
}

int infer(float *input_buffer, float *output_buffer)
{
    // Read input image
    for (int c = 0; c < 3; c++)
    {
        for (int j = 0, HW = HEIGHT * WIDTH; j < HW; ++j)
        {
            input_buffer[c * HW + j] = (static_cast<float>(mPPM.buffer[j * 3 + c]) / mPPM.max - mMean[c]) / mStd[c];
        }
    }

    return buffer;
    util::RGBImageReader reader(input_filename, nvinfer1::Dims3{3, height, width}, {0.485f, 0.456f, 0.406f}, {0.229f, 0.224f, 0.225f});
    reader.read();

    // Preprocess input
    auto input = reader.process();
    cudaMemcpy(buffers[0], input.get(), inputSize, cudaMemcpyHostToDevice);

    // Run inference
    context->execute(1, buffers);

    // Postprocess output
    auto output = std::make_unique<float[]>(width * height);
    cudaMemcpy(output.get(), buffers[1], outputSize, cudaMemcpyDeviceToHost);

    // Write output image
    util::PPM ppm;
    ppm.filename = output_filename;
    ppm.magic = "P5";
    ppm.c = 1;
    ppm.h = height;
    ppm.w = width;
    ppm.max = 255;
    ppm.buffer.resize(width * height);
    for (int i = 0; i < width * height; ++i)
    {
        ppm.buffer[i] = static_cast<uint8_t>(255 * output[i]);
    }
    util::ImageBase image(output_filename, nvinfer1::Dims3{1, height, width});
    image.write();

    return 0;
}

// int destroy_engine()
// {
//     // Release resources
//     cudaFree(buffers[0]);
//     cudaFree(buffers[1]);
//     CONTEXT->destroy();
//     M_ENGINE->destroy();

//     return 0;
// }