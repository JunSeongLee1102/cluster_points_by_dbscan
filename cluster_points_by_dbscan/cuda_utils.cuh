#pragma once
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>
#include "types.cuh"
#include <opencv2/opencv.hpp>


//√‚√≥: https://github.com/xmba15/generic_dbscan/blob/master/cuda/include/dbscan/CudaUtils.cuh
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>


namespace cuda
{
	namespace utils
	{
		cudaError_t warmUpGPU();
	}  // namespace utils
}  // namespace cuda


inline void HandleError(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess) {
		printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
		exit(EXIT_FAILURE);
	}
}


#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

