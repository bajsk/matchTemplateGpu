#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/device/common.hpp>

double launchMatchTemplateGpu(cv::gpu::GpuMat& img, cv::gpu::GpuMat& templ, cv::gpu::GpuMat& result, const dim3 block, const int loop_num);

// use static shared memory
double launchMatchTemplateGpu_withStaticSharedMemory(cv::gpu::GpuMat& img, cv::gpu::GpuMat& templ, cv::gpu::GpuMat& result, const dim3 block, const int loop_num);

// use dynamic shared memory
double launchMatchTemplateGpu_withDynamicSharedMemory(cv::gpu::GpuMat& img, cv::gpu::GpuMat& templ, cv::gpu::GpuMat& result, const dim3 block, const int loop_num);

// use static shared memory with loop unrolling
double launchMatchTemplateGpu_withStaticSharedMemory_withLoopUnrolling(cv::gpu::GpuMat& img, cv::gpu::GpuMat& templ, cv::gpu::GpuMat& result, const dim3 block, const int loop_num);

// use dynamic shared memory with loop unrolling
double launchMatchTemplateGpu_withDynamicSharedMemory_withLoopUnrolling(cv::gpu::GpuMat& img, cv::gpu::GpuMat& templ, cv::gpu::GpuMat& result, const dim3 block, const int loop_num);

