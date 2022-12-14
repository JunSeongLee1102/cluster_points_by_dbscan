#pragma once
#include <filesystem>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "opencv2/core/cuda.hpp"
#include "opencv2/cudafilters.hpp"
#include "opencv2/cudaimgproc.hpp"
#include <opencv2/highgui.hpp>
#include "opencv2/cudaarithm.hpp"
#include <tchar.h>
#include <cmath>
#include <random>
#include <cstdint>
#include <iostream>
#define _USE_MATH_DEFINES
#include <math.h>   
#include <numeric>

#include <chrono>
#include <random>
#include <algorithm>

#include <opencv2/cudafilters.hpp>
#include <opencv2/cudawarping.hpp >

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cuda_utils.cuh"
#include "dbscan.cuh"
#include "types.cuh"

#include <opencv2/core/cuda_stream_accessor.hpp>
#include <thread>
#include <atomic>


using std::shared_ptr;
using std::make_shared;
using std::vector;
using cv::Mat;
using cv::cuda::GpuMat;
using std::string;



extern "C"
{
	void subtract_with_cuda(cv::cuda::GpuMat* src1, cv::cuda::GpuMat* src2, cv::cuda::GpuMat* dst, int block_size_x, int block_size_y, cudaStream_t* stream);
}



struct slope;




class LEDAnalyzer
{
public:

	int thread_index = 0;
	cv::cuda::Stream* cv_stream;
	cudaStream_t* stream;
	//result value
	shared_ptr<OneImgLEDInfo> image_led_info = make_shared<OneImgLEDInfo>();

	//전체 이미지에 선 긋기
	vector<PairPointXY> led_edges;
	
	LEDClusterInfo* led_clusters;

	//inputs and buffers definition		
	//계속 바뀌는 거랑 함수 한번 돌릴 때 한번 쓰는 거는 괜히 헷갈리게 멤버로 갈 필요 없음
	Mat   total_cpu_image_largest_roi_region;
	Mat   cpu_image_largest_roi_region;
	Mat   cpu_comp_gray_image;

	GpuMat gpu_total_gray_image;
	GpuMat gpu_image_largest_roi_region;
	GpuMat gpu_ori_gray_image;






	Mat    cpu_gray_edges;
	GpuMat gpu_gray_edges;








	GpuMat gpu_horizontal_low_sigma;
	GpuMat gpu_horizontal_high_sigma;
	GpuMat gpu_horizontal_dog;

	GpuMat gpu_vertical_low_sigma;
	GpuMat gpu_vertical_high_sigma;
	GpuMat gpu_vertical_dog;

	vector<GpuMat> other_regions;

	vector<slope> pos_horizontal_edges;
	vector<slope> pos_vertical_edges;

	shared_ptr<vector<double>> refined_pos_horizontal_edges = make_shared<vector<double>>();
	shared_ptr<vector<double>> refined_pos_vertical_edges = make_shared<vector<double>>();

	//shared_ptr<double*> horizontal_mean_std = make_shared<double*>(new double[2]);
	//shared_ptr<double*> vertical_mean_std = make_shared<double*>(new double[2]);
	
	double horizontal_mean_std[2];
	double vertical_mean_std[2];
	
	vector<RoiRegion> roi_regions;
	vector<RoiRegion> original_roi_regions;

	int ind_largest_region;
	int ind_largest_roi_region;



	bool process_image(Mat cpu_image, string save_path, bool is_use_history = false, int line_index=0, int row_index = 0, 
					   InformationForWaferCoordinateCalculation info_wafer = InformationForWaferCoordinateCalculation(), int thread_index=0);


	void draw_image(std::string save_path, int draw_option=0);

	bool process_remain_images(string save_path, GpuMat gpu_remain_image, double theat, double interval_x, double interval_y, int draw_option = 0,
								int line_index = 0, int row_index = 0, int pos_y_start = 0, int pos_x_start = 0,
								InformationForWaferCoordinateCalculation info_wafer = InformationForWaferCoordinateCalculation(),
								cudaStream_t* remain_streams = nullptr, cv::cuda::Stream* remain_cv_streams = nullptr);
	
	
	void process_remain_image_wrapper(string save_path, GpuMat gpu_total_image, double theat, double interval_x, double interval_y, int draw_option = 0,
		int line_index = 0, int row_index = 0, int pos_y_start = 0, int pos_x_start = 0,
		InformationForWaferCoordinateCalculation info_wafer = InformationForWaferCoordinateCalculation(),
		cudaStream_t* remain_streams = nullptr, cv::cuda::Stream* remain_cv_streams = nullptr, int ind_roi_region = 0);

	LEDAnalyzer();
	~LEDAnalyzer();
	
	
	void display(Mat image, int x_size, int y_size);
	void display(GpuMat gpu_image, int x_size = 1000, int y_size = 800, cv::cuda::Stream* cv_stream_local=nullptr);
	void display(shared_ptr<cv::Mat> image, int x_size, int y_size);
	void display(shared_ptr<cv::cuda::GpuMat> image, int x_size, int y_size);

	vector<double> global_thetas;
	vector<double> shear_thetas;

	std::atomic<int> slope_counter = 0;
	std::atomic<int> shear_counter = 0;


private:
	double interval_x = 27.233603244357631;
	double interval_y = 17.558903002271464;
	


	double global_theta = 0.0;
	std::atomic<double> shear_theta = 0.0;


	vector<IndexRange> ind_range;

	vector<int> led_range;

	shared_ptr<Dbscan> dbscan;// = make_shared<Dbscan>(1.0, 4.0, 2, cv::cuda::GpuMat(), 4000, ); //for dbscan clustering

	bool is_once_read = false;
	bool is_filter_loaded[2] = { false, false };
	bool is_dbscan = false;
	//Mat 및 GpuMat 정의


	double radius2stdev(double radius);




	vector<double> remove_high_deviation(vector<double> input_array, double sigma, int num_iters = 1);
	
	PairPointXY LEDAnalyzer::inverse_rotate_shear(PairPointXY pair, int pos_y_shift, int pos_x_shift);


    void convert_gray_image_to_edges(GpuMat* gpu_image, GpuMat* gpu_image_return, GpuMat* low_sigma, GpuMat* high_sigma, GpuMat* dog,
		int x_filter_size, int y_filter_size, int x_iter, int y_iter, int grad_or_erode, int do_median_filter, cv::cuda::Stream* cv_local_stream = nullptr);

	void rotate_image(Mat* src, double degree, cv::Point2f base = cv::Point2f(std::numeric_limits<float>::infinity()));

	void gpu_rotate_image(GpuMat* src, double degree,  cv::cuda::Stream* cv_local_stream= nullptr);

	void gpu_shear_image(GpuMat* src, double degree,  cv::cuda::Stream* cv_local_stream = nullptr);


	double calculate_theta_by_dbscan(Mat bound_points, int compressed_x_size, int src_x_size, cv::cuda::Stream* cv_local_stream = nullptr, bool do_gen_dbscan = false);

	vector<slope> extract_horizontal_lines_from_edges(Mat* img_edges, double interval);

	bool correct_image_theta(GpuMat gpu_target_image, int compressed_x_size, int x_filter_size, int y_filter_size, int is_transposed=0, bool is_use_history = false, cv::cuda::Stream * local_cv_stream = nullptr,
		                     bool do_gen_dbscan = false);

	void calculate_mean_std(vector<double> input_array, double* mean_std);

	void LEDAnalyzer::refine_uLED_edge_pos(vector<slope> edge_pos, double filter_out_sigma, vector<double>* refined_edge_pos, int pixel_range);

	void LEDAnalyzer::refine_uLED_edge_pos(vector<slope> edge_pos, double filter_out_sigma, vector<double>* refined_edge_pos, int pixel_range, double interval);






};




