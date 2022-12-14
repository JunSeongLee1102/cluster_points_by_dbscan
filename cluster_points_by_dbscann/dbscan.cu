#pragma once
#include "dbscan.cuh"
#include "cuda_utils.cuh"


using namespace std;
using cv::cuda::GpuMat;
using cv::Mat;


Dbscan::Dbscan(float divide_factor, float eps, int min_points, int buffer_size, cudaStream_t* stream, cv::cuda::Stream* cv_stream)
{
	_stream = stream;
	_cv_stream = cv_stream;
	_eps = eps;
	_min_points = min_points;
	_divide_factor = divide_factor;
	_buffer_size = buffer_size;

	HANDLE_ERROR(cudaMallocAsync((void**)&_dev_neighbor, _buffer_size * _buffer_size * sizeof(int), _stream[0]));
	HANDLE_ERROR(cudaMallocAsync((void**)&_cuda_sample, _buffer_size * sizeof(PointXY), _stream[0]));
	HANDLE_ERROR(cudaMallocAsync((void**)&_gpu_index, _buffer_size * sizeof(PointIndex), _stream[0]));

	cudaMallocHost((void**)&_host_sample, _buffer_size * sizeof(PointXY));
	cudaMallocHost((void**)&_host_index, _buffer_size * sizeof(PointIndex));
	cudaMallocHost((void**)&_host_neighbor, _buffer_size * _buffer_size * sizeof(int));
	cudaMallocHost((void**)&_max_ind_num, 2 * sizeof(int));
	cudaMallocHost((void**)&_index_list, _buffer_size * sizeof(int));

	HANDLE_ERROR(cudaStreamSynchronize(_stream[0]));
}


Dbscan::~Dbscan()
{
	HANDLE_ERROR(cudaFree(_dev_neighbor));

	if (_is_cuda_sample_malloced)
		HANDLE_ERROR(cudaFree(_cuda_sample));

	HANDLE_ERROR(cudaFree(_gpu_index));

	cudaFreeHost(_host_sample);
	cudaFreeHost(_host_index);
	cudaFreeHost(_host_neighbor);
	cudaFreeHost(_max_ind_num);
	cudaFreeHost(_index_list);
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
	_is_cuda_sample_malloced = true;
	_eps = eps;
	_min_points = min_points;
	_divide_factor = divide_factor;
	_num_samples = num_samples;
	_init_block = std::ceil(float(_num_samples) / std::floor(_divide_factor));
	_init_grid = _divide_factor;

	_gpumat_sample.upload(cpu_sample, _cv_stream[0]);
	HANDLE_ERROR(cudaStreamSynchronize(_stream[0]));

	_num_samples = _gpumat_sample.rows;
	float _divide_factor = std::sqrt(_gpumat_sample.rows);
	int step_size = _gpumat_sample.step / sizeof(PointXY);

	if (_num_samples > _buffer_size)
	{
		cout << "Buffer의 크기를 늘려야 합니다.\n";
		system("pause");
	}

	dim3 block(_num_samples);
	dim3 grid(_num_samples);

	HANDLE_ERROR(cudaMemsetAsync(_dev_neighbor, 0, _num_samples * _num_samples * sizeof(int), _stream[0]));
	HANDLE_ERROR(cudaMemsetAsync(_gpu_index, -1, 2 * _num_samples * sizeof(int), _stream[0]));

	gpumatToCudaArray << <_init_block, _init_grid, 0, _stream[0] >> > (reinterpret_cast<PointXY*>(_gpumat_sample.data), _cuda_sample, _num_samples, step_size);

	HANDLE_ERROR(cudaMemcpyAsync(_host_index, _gpu_index, _num_samples * sizeof(PointIndex), cudaMemcpyDeviceToHost, _stream[0]));
}


void Dbscan::PrepareToProcess(int num_samples, float divide_factor, float eps, int min_points, PointXY* cuda_sample)
{
	_is_cuda_sample_malloced = false;
	_eps = eps;
	_min_points = min_points;
	_divide_factor = divide_factor;
	_num_samples = num_samples;
	_init_block = std::ceil(float(_num_samples) / std::floor(_divide_factor));
	_init_grid = _divide_factor;

	if (_num_samples > _buffer_size)
	{
		cout << "Buffer의 크기를 늘려야 합니다.\n";
		system("pause");
	}

	_cuda_sample = cuda_sample;

	HANDLE_ERROR(cudaMemsetAsync(_dev_neighbor, 0, _num_samples * _num_samples * sizeof(int), _stream[0]));
	HANDLE_ERROR(cudaMemsetAsync(_gpu_index, -1, 2 * _num_samples * sizeof(int), _stream[0]));
	HANDLE_ERROR(cudaMemcpyAsync(_host_index, _gpu_index, _num_samples * sizeof(PointIndex), cudaMemcpyDeviceToHost, _stream[0]));
}


void __global__ initializePointIndex(PointIndex* index, int value, int _num_samples)
{
	unsigned int	tid = blockIdx.x * blockDim.x + threadIdx.x;

	if (tid < _num_samples)
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
void __global__ devRegionQuery(PointXY* sample, PointIndex* index, int num, int* neighbors, float _eps, int min_nb)
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

			if (radius < _eps) 
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
	devRegionQuery << <_init_block, _init_grid, 0, _stream[0] >> > (_cuda_sample, _gpu_index, _num_samples, _dev_neighbor, _eps, _min_points);

	HANDLE_ERROR(cudaMemcpyAsync(_host_index, _gpu_index, _num_samples * sizeof(PointIndex), cudaMemcpyDeviceToHost, _stream[0]));
	HANDLE_ERROR(cudaMemcpyAsync(_host_sample, _cuda_sample, _num_samples * sizeof(PointXY), cudaMemcpyDeviceToHost, _stream[0]));
	HANDLE_ERROR(cudaMemcpyAsync(_host_neighbor, _dev_neighbor, _num_samples * _num_samples * sizeof(int), cudaMemcpyDeviceToHost, _stream[0]));
	HANDLE_ERROR(cudaStreamSynchronize(_stream[0]));

	queue<int> expand;
	int cur_cluster = 0;

	for (int i = 0; i < _num_samples; i++) 
	{
		if (_host_index[i].noise >= 0 && _host_index[i].cluster < 1) 
		{
			_host_index[i].cluster = ++cur_cluster;
			int src = i * _num_samples;
			for (int j = 0; j < _num_samples; j++) 
			{
				if (_host_neighbor[src + j]) 
				{
					_host_index[j].cluster = cur_cluster;
					expand.push(j);
				}
			}

			while (!expand.empty()) 
			{/*expand the cluster*/
				if (_host_index[expand.front()].noise >= 0) 
				{
					src = expand.front() * _num_samples;
					for (int j = 0; j < _num_samples; j++) 
					{
						if (_host_neighbor[src + j] && _host_index[j].cluster < 1) 
						{
							_host_index[j].cluster = cur_cluster;
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
	int** cluster_result = new int* [_num_samples]();

	hostAlgorithmDbscan();

	for (int i = 0; i < _num_samples; i++)
		cluster_result[i] = new int[3];

	for (int i = 0; i < _num_samples; i++)
	{
		cluster_result[i][0] = _host_sample[i].x;
		cluster_result[i][1] = _host_sample[i].y;
		cluster_result[i][2] = _host_index[i].cluster;
	}

	return cluster_result;
}



