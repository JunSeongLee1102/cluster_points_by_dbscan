#pragma once
#include "dbscan.cuh"
#include "cuda_utils.cuh"


using namespace std;
using cv::cuda::GpuMat;
using cv::Mat;


Dbscan::Dbscan(float divide_factor, float eps, int min_points, int buffer_size, cudaStream_t* stream, cv::cuda::Stream* cv_stream)
{
	stream_ = stream;
	cv_stream_ = cv_stream;
	eps_ = eps;
	min_points_ = min_points;
	divide_factor_ = divide_factor;
	buffer_size_ = buffer_size;

	HANDLE_ERROR(cudaMallocAsync((void**)&dev_neighbor_, buffer_size_ * buffer_size_ * sizeof(int), stream_[0]));
	HANDLE_ERROR(cudaMallocAsync((void**)&cuda_sample_, buffer_size_ * sizeof(PointXY), stream_[0]));
	HANDLE_ERROR(cudaMallocAsync((void**)&gpu_index_, buffer_size_ * sizeof(PointIndex), stream_[0]));

	cudaMallocHost((void**)&host_sample_, buffer_size_ * sizeof(PointXY));
	cudaMallocHost((void**)&host_index_, buffer_size_ * sizeof(PointIndex));
	cudaMallocHost((void**)&host_neighbor_, buffer_size_ * buffer_size_ * sizeof(int));
	cudaMallocHost((void**)&max_ind_num_, 2 * sizeof(int));
	cudaMallocHost((void**)&index_list_, buffer_size_ * sizeof(int));

	HANDLE_ERROR(cudaStreamSynchronize(stream_[0]));
}


Dbscan::~Dbscan()
{
	HANDLE_ERROR(cudaFree(dev_neighbor_));

	if (is_cuda_sample_malloced_)
		HANDLE_ERROR(cudaFree(cuda_sample_));

	HANDLE_ERROR(cudaFree(gpu_index_));

	cudaFreeHost(host_sample_);
	cudaFreeHost(host_index_);
	cudaFreeHost(host_neighbor_);
	cudaFreeHost(max_ind_num_);
	cudaFreeHost(index_list_);
}


void __global__ gpumatToCudaArray(PointXY* gpumat_sample, PointXY* cuda_sample, int num_samples, int step_size)
{
	unsigned int	tid = blockIdx.x * blockDim.x + threadIdx.x;
	
	if (tid < num_samples)
	{
		unsigned int gpumat_pos = tid * step_size;
		cuda_sample[tid].x = gpumat_sample[gpumat_pos].x;
		cuda_sample[tid].y = gpumat_sample[gpumat_pos].y;
	}
}


void Dbscan::PrepareToProcess(int num_samples, float divide_factor, float eps, int min_points, Mat cpu_sample)
{
	is_cuda_sample_malloced_ = true;
	eps_ = eps;
	min_points_ = min_points;
	divide_factor_ = divide_factor;
	num_samples_ = num_samples;
	init_block_ = std::ceil(float(num_samples_) / std::floor(divide_factor_));
	init_grid_ = divide_factor_;

	gpumat_sample_.upload(cpu_sample, cv_stream_[0]);
	HANDLE_ERROR(cudaStreamSynchronize(stream_[0]));

	num_samples_ = gpumat_sample_.rows;
	float divide_factor_ = std::sqrt(gpumat_sample_.rows);
	int step_size = gpumat_sample_.step / sizeof(PointXY);

	if (num_samples_ > buffer_size_)
	{
		cout << "Buffer의 크기를 늘려야 합니다.\n";
		system("pause");
	}

	dim3 block(num_samples_);
	dim3 grid(num_samples_);

	HANDLE_ERROR(cudaMemsetAsync(dev_neighbor_, 0, num_samples_ * num_samples_ * sizeof(int), stream_[0]));
	HANDLE_ERROR(cudaMemsetAsync(gpu_index_, -1, 2 * num_samples_ * sizeof(int), stream_[0]));

	gpumatToCudaArray << <init_block_, init_grid_, 0, stream_[0] >> > (reinterpret_cast<PointXY*>(gpumat_sample_.data), cuda_sample_, num_samples_, step_size);

	HANDLE_ERROR(cudaMemcpyAsync(host_index_, gpu_index_, num_samples_ * sizeof(PointIndex), cudaMemcpyDeviceToHost, stream_[0]));
}


void Dbscan::PrepareToProcess(int num_samples, float divide_factor, float eps, int min_points, PointXY* cuda_sample)
{
	is_cuda_sample_malloced_ = false;
	eps_ = eps;
	min_points_ = min_points;
	divide_factor_ = divide_factor;
	num_samples_ = num_samples;
	init_block_ = std::ceil(float(num_samples_) / std::floor(divide_factor_));
	init_grid_ = divide_factor_;

	if (num_samples_ > buffer_size_)
	{
		cout << "Buffer의 크기를 늘려야 합니다.\n";
		system("pause");
	}

	cuda_sample_ = cuda_sample;

	HANDLE_ERROR(cudaMemsetAsync(dev_neighbor_, 0, num_samples_ * num_samples_ * sizeof(int), stream_[0]));
	HANDLE_ERROR(cudaMemsetAsync(gpu_index_, -1, 2 * num_samples_ * sizeof(int), stream_[0]));
	HANDLE_ERROR(cudaMemcpyAsync(host_index_, gpu_index_, num_samples_ * sizeof(PointIndex), cudaMemcpyDeviceToHost, stream_[0]));
}


void __global__ initializePointIndex(PointIndex* index, int value, int num_samples_)
{
	unsigned int	tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < num_samples_)
	{
		index[tid].noise = value;
		index[tid].cluster = value;
	}
}


float __device__ devEuclideanDistance(const PointXY& src, const PointXY& dest) 
{

	float res = (src.x - dest.x) * (src.x - dest.x) + (src.y - dest.y) * (src.y - dest.y);

	return sqrt(res);
}

/*to get the total list*/
void __global__ devRegionQuery(PointXY* sample, PointIndex* index, int num, int* neighbors, float eps_, int min_nb)
{
	unsigned int	tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int	line, col, pointer = tid;
	unsigned int	count;

	while (pointer < num * num) 
	{
		line = pointer / num;
		col = pointer % num;
		float radius;
		if (line <= col) 
		{
			radius = devEuclideanDistance(sample[line], sample[col]);

			if (radius < eps_) 
			{
				neighbors[pointer] = 1;
			}
			neighbors[col * num + line] = neighbors[pointer];
		}
		pointer += num;
	}
	__syncthreads();

	pointer = tid;
	while (pointer < num) 
	{
		count = 0;
		line = pointer * num;
		for (int i = 0; i < num; i++)
		{
			if (pointer != i && neighbors[line + i])
			{
				count++;
			}
		}
		if (count >= min_nb) 
		{
			index[pointer].noise++;
		}
		pointer += num;
	}
}

void Dbscan::hostAlgorithmDbscan() 
{
	devRegionQuery << <init_block_, init_grid_, 0, stream_[0] >> > (cuda_sample_, gpu_index_, num_samples_, dev_neighbor_, eps_, min_points_);

	HANDLE_ERROR(cudaMemcpyAsync(host_index_, gpu_index_, num_samples_ * sizeof(PointIndex), cudaMemcpyDeviceToHost, stream_[0]));
	HANDLE_ERROR(cudaMemcpyAsync(host_sample_, cuda_sample_, num_samples_ * sizeof(PointXY), cudaMemcpyDeviceToHost, stream_[0]));
	HANDLE_ERROR(cudaMemcpyAsync(host_neighbor_, dev_neighbor_, num_samples_ * num_samples_ * sizeof(int), cudaMemcpyDeviceToHost, stream_[0]));
	HANDLE_ERROR(cudaStreamSynchronize(stream_[0]));

	queue<int> expand;
	int cur_cluster = 0;

	for (int i = 0; i < num_samples_; i++) 
	{
		if (host_index_[i].noise >= 0 && host_index_[i].cluster < 1) 
		{
			host_index_[i].cluster = ++cur_cluster;
			int src = i * num_samples_;
			for (int j = 0; j < num_samples_; j++) 
			{
				if (host_neighbor_[src + j]) 
				{
					host_index_[j].cluster = cur_cluster;
					expand.push(j);
				}
			}

			while (!expand.empty()) 
			{/*expand the cluster*/
				if (host_index_[expand.front()].noise >= 0) 
				{
					src = expand.front() * num_samples_;
					for (int j = 0; j < num_samples_; j++) 
					{
						if (host_neighbor_[src + j] && host_index_[j].cluster < 1) 
						{
							host_index_[j].cluster = cur_cluster;
							expand.push(j);
						}
					}
				}
				expand.pop();
			}
		}
	}
}



int** Dbscan::Clustering()
{
	int** cluster_result = new int* [num_samples_]();

	hostAlgorithmDbscan();

	for (int i = 0; i < num_samples_; i++)
		cluster_result[i] = new int[3];

	for (int i = 0; i < num_samples_; i++)
	{
		cluster_result[i][0] = host_sample_[i].x;
		cluster_result[i][1] = host_sample_[i].y;
		cluster_result[i][2] = host_index_[i].cluster;
	}

	return cluster_result;
}



