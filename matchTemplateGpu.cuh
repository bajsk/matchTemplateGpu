#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/gpu/device/common.hpp>

double launchMatchTemplateGpu(cv::gpu::GpuMat& img, cv::gpu::GpuMat& templ, cv::gpu::GpuMat& result, const int loop_num);

// use shared memory
double launchMatchTemplateGpu_opt(cv::gpu::GpuMat& img, cv::gpu::GpuMat& templ, cv::gpu::GpuMat& result, const int loop_num);
