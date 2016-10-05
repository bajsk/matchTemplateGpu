#include "matchTemplateGpu.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>

__global__ void matchTemplateGpu
(
    const cv::gpu::PtrStepSz<uchar> img, 
    const cv::gpu::PtrStepSz<uchar> templ, 
    cv::gpu::PtrStepSz<float> result
)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if((x < result.cols) && (y < result.rows)){
        long sum = 0;
        for(int yy = 0; yy < templ.rows; yy++){
            for(int xx = 0; xx < templ.cols; xx++){
                int diff = (img.ptr((y+yy))[x+xx] - templ.ptr(yy)[xx]);
                sum += abs(diff);
            }
        }
        result.ptr(y)[x] = sum;
    }
}

void launchMatchTemplateGpu
(
    cv::gpu::GpuMat& img, 
    cv::gpu::GpuMat& templ, 
    cv::gpu::GpuMat& result
)
{
    cv::gpu::PtrStepSz<uchar> pImg =
        cv::gpu::PtrStepSz<uchar>(img.rows, img.cols * img.channels(), img.ptr<uchar>(), img.step);

    cv::gpu::PtrStepSz<uchar> pDst =
        cv::gpu::PtrStepSz<uchar>(templ.rows, templ.cols * templ.channels(), templ.ptr<uchar>(), templ.step);

    cv::gpu::PtrStepSz<float> pResult =
        cv::gpu::PtrStepSz<float>(result.rows, result.cols * result.channels(), result.ptr<float>(), result.step);

    const dim3 block(64, 2);
    const dim3 grid(cv::gpu::divUp(result.cols, block.x), cv::gpu::divUp(result.rows, block.y));

    matchTemplateGpu<<<grid, block>>>(pImg, pDst, pResult);

   cudaSafeCall(cudaGetLastError());
   cudaSafeCall(cudaDeviceSynchronize());
}

double launchMatchTemplateGpu
(
    cv::gpu::GpuMat& img, 
    cv::gpu::GpuMat& templ, 
    cv::gpu::GpuMat& result, 
    const int loop_num
)
{
    double f = 1000.0f / cv::getTickFrequency();
    int64 start = 0, end = 0;
    double time = 0.0;
    for (int i = 0; i <= loop_num; i++){
        start = cv::getTickCount();
        launchMatchTemplateGpu(img, templ, result);
        end = cv::getTickCount();
        time += (i > 0) ? ((end - start) * f) : 0;
    }
    time /= loop_num;

    return time;
}

// use static shared memory
__global__ void matchTemplateGpu_withStaticSharedMemory
(
 const cv::gpu::PtrStepSz<uchar> img,
 const cv::gpu::PtrStepSz<uchar> templ,
 cv::gpu::PtrStepSz<float> result
 )
{
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;

  __shared__ uchar temp[32 * 32];

  if(threadIdx.x == 0){
    for(int yy = 0; yy < templ.rows; yy++){
      for(int xx = 0; xx < templ.cols; xx++){
	temp[yy*templ.cols+xx] = templ.ptr(yy)[xx];
      }
    }
  }
  __syncthreads();

  if((x < result.cols) && (y < result.rows)){
    long sum = 0;
    for(int yy = 0; yy < templ.rows; yy++){
      for(int xx = 0; xx < templ.cols; xx++){
	int diff = (img.ptr((y+yy))[x+xx] - temp[yy*templ.cols+xx]);
	sum += abs(diff);
      }
    }
    result.ptr(y)[x] = sum;
  }
}

// use static shared memory
void launchMatchTemplateGpu_withStaticSharedMemory
(
 cv::gpu::GpuMat& img,
 cv::gpu::GpuMat& templ,
 cv::gpu::GpuMat& result
 )
{
  cv::gpu::PtrStepSz<uchar> pImg =
    cv::gpu::PtrStepSz<uchar>(img.rows, img.cols * img.channels(), img.ptr<uchar>(), img.step);

  cv::gpu::PtrStepSz<uchar> pDst =
    cv::gpu::PtrStepSz<uchar>(templ.rows, templ.cols * templ.channels(), templ.ptr<uchar>(), templ.step);

  cv::gpu::PtrStepSz<float> pResult =
    cv::gpu::PtrStepSz<float>(result.rows, result.cols * result.channels(), result.ptr<float>(), result.step);

  const dim3 block(64, 2);
  const dim3 grid(cv::gpu::divUp(result.cols, block.x), cv::gpu::divUp(result.rows, block.y));

  matchTemplateGpu_withStaticSharedMemory<<<grid, block>>>(pImg, pDst, pResult);

  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaDeviceSynchronize());
}

// use static shared memory
double launchMatchTemplateGpu_withStaticSharedMemory
(
 cv::gpu::GpuMat& img, 
 cv::gpu::GpuMat& templ, 
 cv::gpu::GpuMat& result, 
    const int loop_num
 )
{
  double f = 1000.0f / cv::getTickFrequency();
  int64 start = 0, end = 0;
  double time = 0.0;
  for (int i = 0; i <= loop_num; i++){
    start = cv::getTickCount();
    launchMatchTemplateGpu_withStaticSharedMemory(img, templ, result);
    end = cv::getTickCount();
    time += (i > 0) ? ((end - start) * f) : 0;
  }
  time /= loop_num;

  return time;
}

// use dynamic shared memory
__global__ void matchTemplateGpu_withDynamicSharedMemory
(
 const cv::gpu::PtrStepSz<uchar> img,
 const cv::gpu::PtrStepSz<uchar> templ,
 cv::gpu::PtrStepSz<float> result
 )
{
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;

  extern __shared__ uchar temp[];

  if(threadIdx.x == 0){
    for(int yy = 0; yy < templ.rows; yy++){
      for(int xx = 0; xx < templ.cols; xx++){
	temp[yy*templ.cols+xx] = templ.ptr(yy)[xx];
      }
    }
  }
  __syncthreads();

  if((x < result.cols) && (y < result.rows)){
    long sum = 0;
    for(int yy = 0; yy < templ.rows; yy++){
      for(int xx = 0; xx < templ.cols; xx++){
	int diff = (img.ptr((y+yy))[x+xx] - temp[yy*templ.cols+xx]);
	sum += abs(diff);
      }
    }
    result.ptr(y)[x] = sum;
  }
}

// use dynamic shared memory
void launchMatchTemplateGpu_withDynamicSharedMemory
(
 cv::gpu::GpuMat& img,
 cv::gpu::GpuMat& templ,
 cv::gpu::GpuMat& result
 )
{
  cv::gpu::PtrStepSz<uchar> pImg =
    cv::gpu::PtrStepSz<uchar>(img.rows, img.cols * img.channels(), img.ptr<uchar>(), img.step);

  cv::gpu::PtrStepSz<uchar> pDst =
    cv::gpu::PtrStepSz<uchar>(templ.rows, templ.cols * templ.channels(), templ.ptr<uchar>(), templ.step);

  cv::gpu::PtrStepSz<float> pResult =
    cv::gpu::PtrStepSz<float>(result.rows, result.cols * result.channels(), result.ptr<float>(), result.step);

  const dim3 block(64, 2);
  const dim3 grid(cv::gpu::divUp(result.cols, block.x), cv::gpu::divUp(result.rows, block.y));
  const size_t shared_mem_size = templ.cols*templ.rows*sizeof(uchar);

  matchTemplateGpu_withDynamicSharedMemory<<<grid, block, shared_mem_size>>>(pImg, pDst, pResult);

  cudaSafeCall(cudaGetLastError());
  cudaSafeCall(cudaDeviceSynchronize());
}

// use dynamic shared memory
double launchMatchTemplateGpu_withDynamicSharedMemory
(
 cv::gpu::GpuMat& img, 
 cv::gpu::GpuMat& templ, 
 cv::gpu::GpuMat& result, 
    const int loop_num
 )
{
  double f = 1000.0f / cv::getTickFrequency();
  int64 start = 0, end = 0;
  double time = 0.0;
  for (int i = 0; i <= loop_num; i++){
    start = cv::getTickCount();
    launchMatchTemplateGpu_withDynamicSharedMemory(img, templ, result);
    end = cv::getTickCount();
    time += (i > 0) ? ((end - start) * f) : 0;
  }
  time /= loop_num;

  return time;
}
