#include "matchTemplateCpu.h"
#include "matchTemplateGpu.cuh"
#include "utility.h"

#include <iostream>

int main(int argc, char *argv[])
{
    const int loop_num = 5;
    double time;

    cv::Mat img(sz1080p, CV_8UC1, cv::Scalar(0));
    cv::Mat templ(cv::Size(32, 32), CV_8UC1, cv::Scalar(255));
    cv::Size corrSize(img.cols - templ.cols + 1, img.rows - templ.rows + 1);
    cv::Mat result(corrSize, CV_32FC1, cv::Scalar(0.0f));
    cv::Mat result_cv(corrSize, CV_32FC1, cv::Scalar(0.0f));

#ifdef VALIDATION

    // Naive Implementation
    time = launchMatchTemplateCpu(img, templ, result, loop_num);
    std::cout << "Naive: " << time << " ms." << std::endl;

#endif

    cv::gpu::GpuMat d_img(img);
    cv::gpu::GpuMat d_templ(templ);
    cv::gpu::GpuMat d_result(corrSize, CV_32FC1, cv::Scalar(0.0f));
    cv::gpu::GpuMat d_result2(corrSize, CV_32FC1, cv::Scalar(0.0f));
    cv::gpu::GpuMat d_result3(corrSize, CV_32FC1, cv::Scalar(0.0f));
    cv::gpu::GpuMat d_result4(corrSize, CV_32FC1, cv::Scalar(0.0f));
    cv::gpu::GpuMat d_result5(corrSize, CV_32FC1, cv::Scalar(0.0f));
    cv::gpu::GpuMat d_result6(corrSize, CV_32FC1, cv::Scalar(0.0f));
    cv::gpu::GpuMat d_result7(corrSize, CV_32FC1, cv::Scalar(0.0f));

    const dim3 block = dim3(64, 2);

    // CUDA Implementation
    time = launchMatchTemplateGpu(d_img, d_templ, d_result, block, loop_num);
    std::cout << "CUDA: " << time << " ms." << std::endl;

    // CUDA Implementation (static shared memory)
    time = launchMatchTemplateGpu_withStaticSharedMemory(d_img, d_templ, d_result2, block, loop_num);
    std::cout << "CUDA(withStaticSharedMemory): " << time << " ms." << std::endl;

    // CUDA Implementation (dynamic shared memory)
    time = launchMatchTemplateGpu_withDynamicSharedMemory(d_img, d_templ, d_result3, block, loop_num);
    std::cout << "CUDA(withDynamicSharedMemory): " << time << " ms." << std::endl;

    // CUDA Implementation (static shared memory with loop unrolling)
    time = launchMatchTemplateGpu_withStaticSharedMemory_withLoopUnrolling(d_img, d_templ, d_result4, block, loop_num);
    std::cout << "CUDA(withStaticSharedMemory_withLoopUnrolling): " << time << " ms." << std::endl;

    // CUDA Implementation (dynamic shared memory with loop unrolling)
    time = launchMatchTemplateGpu_withDynamicSharedMemory_withLoopUnrolling(d_img, d_templ, d_result5, block, loop_num);
    std::cout << "CUDA(withDynamicSharedMemory_withLoopUnrolling): " << time << " ms." << std::endl;

    const dim3 block2 = dim3(128, 1);

    // CUDA Implementation (dynamic shared memory with loop unrolling and different block size)
    time = launchMatchTemplateGpu_withDynamicSharedMemory_withLoopUnrolling(d_img, d_templ, d_result6, block2, loop_num);
    std::cout << "CUDA(withDynamicSharedMemory_withLoopUnrolling_blockSize(128x1): " << time << " ms." << std::endl;

    // CUDA Implementation (dynamic shared memory with loop unrolling and read only cache)
    time = launchMatchTemplateGpu_withDynamicSharedMemory_withLoopUnrolling_readOnlyCache(d_img, d_templ, d_result7, block2, loop_num);
    std::cout << "CUDA(withDynamicSharedMemory_withLoopUnrolling_blockSize_readOnlyCache(128x1): " << time << " ms." << std::endl;

    std::cout << std::endl;

    // Verification
    verify(d_result, d_result2);
    verify(d_result, d_result3);
    verify(d_result, d_result4);
    verify(d_result, d_result5);
    verify(d_result, d_result6);
    verify(d_result, d_result7);

    return 0;
}
