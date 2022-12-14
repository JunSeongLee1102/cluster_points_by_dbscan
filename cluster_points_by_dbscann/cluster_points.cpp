#include "cluster_points.h"


using cv::Mat;
using cv::cuda::GpuMat;


void display(Mat image, int x_size = 1000, int y_size = 1000)
{
	Mat vis_image;
	cv::resize(image, vis_image, cv::Size(x_size, y_size), 0, 0, cv::INTER_AREA);
	cv::imshow("Sample", vis_image);
	cv::waitKey();
}


void display(GpuMat gpu_image, int x_size = 1000, int y_size = 1000, cv::cuda::Stream _cv_stream = cv::cuda::Stream())
{
	Mat vis_image;
	gpu_image.download(vis_image, _cv_stream);
	_cv_stream.waitForCompletion();

	cv::resize(vis_image, vis_image, cv::Size(x_size, y_size), 0, 0, cv::INTER_AREA);
	cv::imshow("Sample", vis_image);
	cv::waitKey();
}


ClusterPoints::ClusterPoints()
{
}


void ClusterPoints::initialize(bool use_gpumat_find_nonzero, bool draw_result)
{
	_use_gpumat_find_nonzero = use_gpumat_find_nonzero;
	_draw_result = draw_result;

	//병렬화 위한 cudaStream 생성
	//cv::cuda::Stream 생성 및 해당 _stream과 같은 CUDA에서의 cudaStream_t get
	_cv_stream = new cv::cuda::Stream();
	_stream = new cudaStream_t();
	_stream[0] = cv::cuda::StreamAccessor::getStream(_cv_stream[0]);

	//clustering을 위한 _dbscan 구조체 생성
	_dbscan = new Dbscan(32.0, 4.0, 5, 10000, _stream, _cv_stream);

	//사각형들 그려진 이미지
	_original_sample_img = cv::Mat(5000, 5000, CV_8UC1);
	_original_sample_img.setTo(0);

	//이미지에 사각형들 추가
	//사각형들 중심 간 거리
	int interval = 1600;
	for (int xi = 0; xi < 3; xi++)
	{
		for (int yj = 0; yj < 3; yj++)
		{
			cv::rectangle(_original_sample_img, cv::Point(800 + xi * interval, 800 + yj * interval), cv::Point(930 + xi * interval, 930 + yj * interval), cv::Scalar(255), -1);
		}
	}

	GpuMat gpu_sample_img;
	gpu_sample_img.upload(_original_sample_img, _cv_stream[0]);
	_cv_stream[0].waitForCompletion();

	if (_use_gpumat_find_nonzero)
		_gpumat_find_nonzero = new GpuMatFindNonzero(_original_sample_img.rows, _original_sample_img.cols, gpu_sample_img.step, 10000, _stream);

	if (_draw_result)
	{
		std::string save_image_name = "saved_images/original_rects";
		save_image_name += ".jpg";
		cv::imwrite(save_image_name, _original_sample_img);
	}
}


ClusterPoints::~ClusterPoints()
{
	if (_use_gpumat_find_nonzero)
		delete _gpumat_find_nonzero;
	delete _cv_stream;
	delete _stream;
	delete _dbscan;
}


void ClusterPoints::clusterRectsInImage(int save_image_index)
{
	Mat sample_img;
	GpuMat gpu_sample_img;
	cv::Ptr<cv::cuda::CannyEdgeDetector> canny = cv::cuda::createCannyEdgeDetector(70.0, 255.0);
	Mat sample_img_nonzeros;

	int num_nonzeros = 0;
	int** cluster_result;
	int** cluster_colors;
	int num_clusters = 0;

	gpu_sample_img.upload(_original_sample_img, _cv_stream[0]);
	canny->detect(gpu_sample_img, gpu_sample_img, _cv_stream[0]);

	if (_use_gpumat_find_nonzero)
	{
		_gpumat_find_nonzero->findNonzero(gpu_sample_img);
		_dbscan->PrepareToProcess(_gpumat_find_nonzero->_num_nonzeros, 32.0, 4.0, 5, (_gpumat_find_nonzero->_nonzero_xy_coords));
		num_nonzeros = _gpumat_find_nonzero->_num_nonzeros;
	}
	else
	{
		gpu_sample_img.download(sample_img, _cv_stream[0]);
		_cv_stream[0].waitForCompletion();
		cv::findNonZero(sample_img, sample_img_nonzeros);
		num_nonzeros = sample_img_nonzeros.rows;
		_dbscan->PrepareToProcess(sample_img_nonzeros.rows, 32.0, 4.0, 5, sample_img_nonzeros);

	}

	cluster_result = _dbscan->Clustering();

	for (int i = 0; i < num_nonzeros; i++)	
		num_clusters = max(cluster_result[i][2], num_clusters);
	
	num_clusters += 2;

	if (_draw_result)
	{
		cluster_colors = new int* [num_clusters];
		int seed1 = rand() % 255;
		int seed2 = rand() % 255;
		int seed3 = rand() % 255;

		for (int i = 0; i < num_clusters; i++)
		{
			cluster_colors[i] = new int[3];
			cluster_colors[i][0] = (seed1 + i * 75) % 255;
			cluster_colors[i][1] = (seed2 + i * 41) % 255;
			cluster_colors[i][2] = (seed3 + i * 81) % 255;
		}

		//draw clusters -->  first make rectangles enclosing each cluster and draw all of them
		sample_img = cv::Mat(gpu_sample_img.rows, gpu_sample_img.cols, CV_8UC3, cv::Scalar(0, 0, 0));

		std::vector<cv::Rect> clustered_rects = std::vector<cv::Rect>(num_clusters, cv::Rect(1000000, 1000000, 0, 0));
		for (int i = 0; i < num_nonzeros; i++)
		{
			clustered_rects[cluster_result[i][2]].x = min(cluster_result[i][0], clustered_rects[cluster_result[i][2]].x);
			clustered_rects[cluster_result[i][2]].y = min(cluster_result[i][1], clustered_rects[cluster_result[i][2]].y);
			clustered_rects[cluster_result[i][2]].width = max(cluster_result[i][0], clustered_rects[cluster_result[i][2]].width);
			clustered_rects[cluster_result[i][2]].height = max(cluster_result[i][1], clustered_rects[cluster_result[i][2]].height);
		}

		for (int i = 0; i < num_clusters; i++)
		{
			clustered_rects[i].width -= clustered_rects[i].x;
			clustered_rects[i].height -= clustered_rects[i].y;
			cv::Scalar color(cluster_colors[i][0], cluster_colors[i][1], cluster_colors[i][2]);
			cv::rectangle(sample_img, clustered_rects[i], color, -1);
		}

		//display(sample_img, 1000, 1000);
		std::string save_image_name = "saved_images/clustered_rects" + std::to_string(save_image_index) + ".jpg";
		cv::imwrite(save_image_name, sample_img);

		for (int i = 0; i < num_clusters; i++)
			delete[] cluster_colors[i];

		delete[] cluster_colors;
	}

	for (int i = 0; i < sample_img_nonzeros.rows; i++)
		delete[] cluster_result[i];
	delete[] cluster_result;
}