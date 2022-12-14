#include "types.cuh"


NonzeroCoords::NonzeroCoords(int num_points, cudaStream_t* stream)
{
	_stream = stream;
	_num_points = num_points;
	HANDLE_ERROR(cudaMallocAsync((void**)&_coords, _num_points * sizeof(PointXY), _stream[0]));
	HANDLE_ERROR(cudaStreamSynchronize(_stream[0]));
}

NonzeroCoords::~NonzeroCoords()
{
	HANDLE_ERROR(cudaFreeAsync(_coords, _stream[0]));
	HANDLE_ERROR(cudaStreamSynchronize(_stream[0]));
}


