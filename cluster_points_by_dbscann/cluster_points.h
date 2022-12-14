#pragma once
#include "opencv2/opencv.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include <opencv2/highgui.hpp>
#include "opencv2/cudaarithm.hpp"
#include <opencv2/core/cuda_stream_accessor.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "dbscan.cuh"
#include "types.cuh"
#include <chrono>

#include "gpumat_find_nonzero.cuh"


class ClusterPoints
{
public:
	ClusterPoints();
	void initialize(bool use_gpumat_find_nonzero = true, bool draw_result = false);
	~ClusterPoints();
	void clusterRectsInImage(int save_image_index);
private:
	bool _use_gpumat_find_nonzero;
	bool _draw_result;
	Dbscan* _dbscan;
	cv::cuda::Stream* _cv_stream;
	cudaStream_t* _stream;

	GpuMatFindNonzero* _gpumat_find_nonzero;

	Mat _original_sample_img;
	GpuMat _gpu_original_sample_img;
};