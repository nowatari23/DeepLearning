/* -*- mode:c++; coding:utf-8-ws-dos; tab-width:4 -*- ==================== */
/* -----------------------------------------------------------------------
 * $Id: NeuralNet.cpp 2720 2018-01-02 21:21:06+09:00 nowatari $
 * ======================================================================= */

#pragma warning(disable:4819)

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string.h>

/*======================================================================
 * Affine変換
 *======================================================================*/
static __global__ void AffineForwardKernel(double *pOutput,
                                           const double *pInput,
                                           const double *pWeight,
                                           int inputNum)
{
  int o = threadIdx.x;

  for (int i = 0; i < inputNum; ++i)
    pOutput[o] += pInput[i] * pWeight[inputNum * o + i];
}

void AffineForward(double *pOutput,
                   const double *pInput,
                   const double *pWeight,
                   int inputNum,
                   int outputNum)
{
  double *pDevInput = 0;
  double *pDevWeight = 0;
  double *pDevOutput = 0;
  cudaError_t cudaStatus;

  do{
    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess)
    {
      fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
      break;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&pDevOutput, outputNum * sizeof(double));
    if (cudaStatus != cudaSuccess)
    {
      fprintf(stderr, "cudaMalloc failed!");
      break;
    }

    cudaStatus = cudaMalloc((void**)&pDevInput, inputNum * sizeof(double));
    if (cudaStatus != cudaSuccess)
    {
      fprintf(stderr, "cudaMalloc failed!");
      break;
    }

    cudaStatus = cudaMalloc((void**)&pDevWeight, outputNum * inputNum *sizeof(double));
    if (cudaStatus != cudaSuccess)
    {
      fprintf(stderr, "cudaMalloc failed!");
      break;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(pDevInput,
                            pInput,
                            inputNum * sizeof(double),
                            cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
      fprintf(stderr, "cudaMemcpy failed!");
      break;
    }

    cudaStatus = cudaMemcpy(pDevWeight,
                            pWeight,
                            outputNum * inputNum * sizeof(double),
                            cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
      fprintf(stderr, "cudaMemcpy failed!");
      break;
    }

    memset(pOutput, 0, outputNum * sizeof(double));

    cudaStatus = cudaMemcpy(pDevOutput,
                            pOutput,
                            outputNum * sizeof(double),
                            cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess)
    {
      fprintf(stderr, "cudaMemcpy failed!");
      break;
    }

    // Launch a kernel on the GPU with one thread for each element.
    AffineForwardKernel<<<1, outputNum>>>(pDevOutput, pDevInput, pDevWeight, inputNum);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess)
    {
      fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
      break;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
      fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
      break;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(pOutput,
                            pDevOutput,
                            outputNum * sizeof(double),
                            cudaMemcpyDeviceToHost);

    if (cudaStatus != cudaSuccess)
    {
      fprintf(stderr, "cudaMemcpy failed!");
      break;
    }
  }while(0);

  cudaFree(pDevOutput);
  cudaFree(pDevInput);
  cudaFree(pDevWeight);
}
