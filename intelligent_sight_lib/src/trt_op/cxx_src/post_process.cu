#include "trt.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>

// input tensor shape (1, 32, 8400)
// 32: 4(xywh) + 18(class) + 10(kpnt)
// output shape (1, 8400, 16)
__global__ void transform_results(float *input_buffer, float *output_buffer)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (x < 8400)
    {
        for (int i = 0; i < 4; i++)
        {
            output_buffer[x * 32 + i] = input_buffer[i * 8400 + x];
        }
        float max_score = input_buffer[4 * 8400 + x];
        int cls = 0;
        for (int i = 5; i < 22; i++)
        {
            if (input_buffer[i * 8400 + x] > max_score)
            {
                max_score = input_buffer[i * 8400 + x];
                cls = i - 4;
            }
        }
        output_buffer[x * 32 + 4] = max_score;
        output_buffer[x * 32 + 5] = (float)cls;

        for (int i = 22; i < 32; i++)
        {
            output_buffer[x * 32 + i] = input_buffer[i * 8400 + x];
        }
    }
}

// input tensor shape (1, 8400, 16)
// 16: 4(xywh) + 1(score) + 1(cls) + 10(kpnt)
// output shape (MAX_DETECTION, 16)
__global__ void nms(float *input_buffer, float *output_buffer)
{
}

PostProcess::~PostProcess()
{
    cudaFree(this->transformed);
    cudaFree(this->indices);
}

uint16_t PostProcess::init()
{
    check_status(cudaMalloc(&this->transformed, 8400 * 16 * sizeof(float)));
    check_status(cudaMalloc(&this->indices, 8400 * sizeof(int)));
    this->d_indices = thrust::device_ptr<int>(this->indices);
    this->d_transformed = thrust::device_ptr<float>(this->transformed);
}

// input buffer (1, 32, 8400)
// output buffer (MAX_DETECTION, 16)
uint16_t PostProcess::post_process(float *input_buffer, float *output_buffer)
{
    dim3 threads_pre_block(48);
    dim3 blocks(175);
    transform_results<<<blocks, threads_pre_block>>>(input_buffer, this->transformed);
    check_status(cudaDeviceSynchronize());
    thrust::sequence(this->d_indices, this->d_indices + 8400);
    thrust::sort(this->d_indices, this->d_indices + 8400, [this] __device__(int a, int b)
                 { return this->d_transformed[a * 32 + 4] > this->d_transformed[b * 32 + 4]; });

    return (uint16_t)cudaSuccess;
}

PostProcess *POSTPROCESS;

uint16_t postprocess_init()
{
    POSTPROCESS = new PostProcess();
    check_status(POSTPROCESS->init());
    return (uint16_t)cudaSuccess;
}

// input buffer (1, 32, 8400)
// output buffer (MAX_DETECTION, 16)
uint16_t postprocess(float *input_buffer, float *output_buffer)
{
    check_status(POSTPROCESS->post_process(input_buffer, output_buffer));
    return (uint16_t)cudaSuccess;
}