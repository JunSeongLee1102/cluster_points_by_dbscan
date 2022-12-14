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
	int _num_samples;
	float _eps;
	int _min_points;
	GpuMat _gpumat_sample;
	int* _index_list;
	int* _max_ind_num;
	float _divide_factor;
	int _buffer_size;
	int _init_grid;
	int _init_block;

	cudaStream_t* _stream;
	cv::cuda::Stream* _cv_stream;

	Dbscan(float divide_factor = 5.0, float eps = 4.0, int min_points = 5, int buffer_size = 1500, cudaStream_t* stream = nullptr, cv::cuda::Stream* cv_stream = nullptr);
	~Dbscan();

	void PrepareToProcess(int num_samples, float divide_factor = 5.0, float eps = 4.0, int min_points = 5, Mat cpu_sample = Mat());
	void PrepareToProcess(int num_samples, float divide_factor, float eps, int min_points, PointXY* gpu_sample);
	int** Clustering();

private:
	PointXY* _host_sample;
	PointIndex* _host_index;
	PointXY* _cuda_sample;
	bool _is_cuda_sample_malloced = false;
	PointIndex* _gpu_index;
	int* _dev_neighbor;
	int* _host_neighbor;

	void hostAlgorithmDbscan();
};

