#pragma once
#include "cublas_v2.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>
#include <windows.h>
#include <math.h>
#include <queue>
#include "types.cuh"
#include "opencv2/opencv.hpp"
#include "opencv2/core/cuda.hpp"
#include <vector>
#include "cuda_runtime.h"
#include "device_functions.h"
#include <thrust/device_vector.h>


//Reference: https://github.com/a0165897/dbscan-cuda/tree/master/dbscanWithCuda


using cv::Mat;
using cv::cuda::GpuMat;


class Dbscan
{
public:
	int num_samples_;
	float eps_;
	int min_points_;
	GpuMat gpumat_sample_;
	int* index_list_;
	int* max_ind_num_;
	float divide_factor_;
	int buffer_size_;
	int init_grid_;
	int init_block_;

	cudaStream_t* stream_;
	cv::cuda::Stream* cv_stream_;

	Dbscan(float divide_factor = 5.0, float eps = 4.0, int min_points = 5, int buffer_size = 1500, cudaStream_t* stream = nullptr, cv::cuda::Stream* cv_stream = nullptr);
	~Dbscan();

	void PrepareToProcess(int num_samples, float divide_factor = 5.0, float eps = 4.0, int min_points = 5, Mat cpu_sample = Mat());
	void PrepareToProcess(int num_samples, float divide_factor, float eps, int min_points, PointXY* gpu_sample);
	int** Clustering();

private:
	PointXY* host_sample_;
	PointIndex* host_index_;
	PointXY* cuda_sample_;
	bool is_cuda_sample_malloced_ = false;
	PointIndex* gpu_index_;
	int* dev_neighbor_;
	int* host_neighbor_;

	void hostAlgorithmDbscan();
};

