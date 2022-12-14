#include "gpumat_find_nonzero.cuh"
#include <cmath>


__global__
void kernelFindNonzero(unsigned char* gpu_image_data, int rows, int cols, int step,
	PointXY* nonzero_xy_coords, int* gpu_num_nonzeros)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y; //Row number
	int j = blockIdx.x * blockDim.x + threadIdx.x; //Column number

	int cur_ind = 0;
	//이미지 범위 내의 인덱스 --> 0이 아닐 경우 nonzero에 저장
	//atomicAdd를 통한 중복 저장이나 메모리 동시 엑세스 방지
	if (i < rows && j < cols)
	{
		if (gpu_image_data[i * step + j] != unsigned char(0))
		{
			cur_ind = atomicAdd(&gpu_num_nonzeros[0], 1);
			nonzero_xy_coords[cur_ind].x = j;
			nonzero_xy_coords[cur_ind].y = i;
		}
	}
}


GpuMatFindNonzero::GpuMatFindNonzero(int rows, int cols, int step, int max_num_nonzeros, cudaStream_t* stream)
{
	_rows = rows;
	_cols = cols;
	_step = step;
	_max_num_nonzeros = max_num_nonzeros;

	//consider warp thread size
	_row_block_size = int(rows / 32 + 1);
	_col_block_size = int(cols / 32 + 1);

	_grid = dim3(_col_block_size, _row_block_size);
	_block = dim3(32, 32);

	_cpu_num_nonzeros = new int[1];
	_cpu_num_nonzeros[0] = 0;

	_stream = stream;

	HANDLE_ERROR(cudaMallocAsync((void**)&_gpu_num_nonzeros, sizeof(int), _stream[0]));
	HANDLE_ERROR(cudaMemsetAsync(_gpu_num_nonzeros, 0, sizeof(int), _stream[0]));
	HANDLE_ERROR(cudaMallocAsync((void**)&_nonzero_xy_coords, 2 * _max_num_nonzeros * sizeof(int), _stream[0]));

	HANDLE_ERROR(cudaStreamSynchronize(_stream[0]));
}


GpuMatFindNonzero::~GpuMatFindNonzero()
{
	HANDLE_ERROR(cudaFreeAsync(_gpu_num_nonzeros, _stream[0]));
	HANDLE_ERROR(cudaFreeAsync(_nonzero_xy_coords, _stream[0]));

	delete[] _cpu_num_nonzeros;


	HANDLE_ERROR(cudaStreamSynchronize(_stream[0]));
}


void GpuMatFindNonzero::findNonzero(cv::cuda::GpuMat gpu_image)
{
	HANDLE_ERROR(cudaMemsetAsync(_gpu_num_nonzeros, 0, sizeof(int), _stream[0]));

	kernelFindNonzero << <_grid, _block, 0, _stream[0] >> > (reinterpret_cast<unsigned char*>(gpu_image.data), _rows,
		_cols, _step, _nonzero_xy_coords, _gpu_num_nonzeros);

	HANDLE_ERROR(cudaMemcpyAsync(&_cpu_num_nonzeros[0], &_gpu_num_nonzeros[0], sizeof(int),
		cudaMemcpyDeviceToHost, _stream[0]));

	HANDLE_ERROR(cudaStreamSynchronize(_stream[0]));

	_num_nonzeros = _cpu_num_nonzeros[0];
	return;
}