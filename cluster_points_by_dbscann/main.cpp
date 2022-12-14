#pragma once
#define CUDA_API_PER_THREAD_DEFAULT_STREAM

#include <fstream>
#include <thread>
#include <iostream>
#include <chrono>
#include "cluster_points.h"
#include <sys/stat.h>
#include <sys/types.h>

using namespace std;
using std::cout;
using std::endl;
using std::string;


//Parameters to control calculation modes and number of processing images
const int num_trials = 1;
const int num_threads = 6;
const int num_total_images = 72;
bool use_gpumat_find_nonzero = true;
bool draw_result = false;


void clusterRectsInImageWrapper(ClusterPoints* cluster_points, std::vector<int> save_image_indices)
{
	for (int i = 0; i < save_image_indices.size(); i++)
		cluster_points->clusterRectsInImage(save_image_indices[i]);
}

//이미지 저장 디렉토리 유무 확인하고 없으면 만들기
void makeImgSaveDirectory(bool draw_result)
{
	if (draw_result)
	{
		struct stat info;
		int check_directory_creation = 0;
		if (stat("saved_images", &info) != 0)
		{
			std::string dir_name = "saved_images";
			std::wstring widestr_dir_name = std::wstring(dir_name.begin(), dir_name.end());
			const wchar_t* wchar_dir_name = widestr_dir_name.c_str();
			printf("cannot access %s, create directory.\n", "saved_images");
			check_directory_creation = _wmkdir(wchar_dir_name);
			if (!check_directory_creation)
				printf("Directory created\n");
			else 
			{
				printf("Unable to create directory\n");
				exit(1);
			}
		}
	}
}

void main()
{
	makeImgSaveDirectory(draw_result);

	//더 빠른 처리를 위해서 pinned memory 사용
	cv::Mat::setDefaultAllocator(cv::cuda::HostMem::getAllocator(cv::cuda::HostMem::AllocType::PAGE_LOCKED));

	ClusterPoints* arr_cluster_points = new ClusterPoints[num_threads]();
	for (int i = 0; i < num_threads; i++)
		arr_cluster_points[i].initialize(use_gpumat_find_nonzero, draw_result);

	std::vector<std::thread> thread_cluster_points;

	std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

	for (int ti = 0; ti < num_threads; ti++)
	{
		std::vector<int> image_indices;
		for (int image_index = 0; image_index < (num_total_images) / num_threads; image_index++)
		{
			image_indices.push_back(image_index + ti * num_total_images / num_threads);
		}
		thread_cluster_points.push_back(std::thread(clusterRectsInImageWrapper, &arr_cluster_points[ti], image_indices));
	}

	for (int i = 0; i < num_threads; i++)
		thread_cluster_points[i].join();

	std::chrono::duration<double>sec = std::chrono::system_clock::now() - start;
	std::cout << "총 이미지 처리 시간: " << sec.count() << "seconds" << std::endl;

	double sec_per_image = sec.count() / num_total_images;
	std::cout << "이미지 당 평균 처리 시간 : " << sec_per_image * 1000.0 << " milli seconds" << std::endl;

	delete[] arr_cluster_points;
	thread_cluster_points.clear();	

	return;
}