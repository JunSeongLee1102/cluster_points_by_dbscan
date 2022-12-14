#pragma once
#include <thrust/device_vector.h>
#include <vector>
#include <numeric>
#include "opencv2/opencv.hpp"
#include "opencv2/core/cuda.hpp"
#include "cuda_utils.cuh"


struct PointXY
{
	int x = 0;
	int y = 0;
};


struct PointIndex 
{
	int noise;
	int cluster;
};


class NonzeroCoords
{
public:
	NonzeroCoords(int num_points, cudaStream_t* stream);
	~NonzeroCoords();


	int _num_points = 0;
	PointXY* _coords;
private:
	cudaStream_t* _stream;
};


namespace clustering
{
	namespace cuda
	{
		enum class NodeType : int { CORE, NOISE };

		struct Node {
			__host__ __device__ Node()
				: type(NodeType::NOISE)
				, numNeighbors(0)
				, visited(false)
			{
			}

			NodeType type;
			int numNeighbors;
			char visited;
		};

		struct Graph {
			thrust::device_vector<Node> nodes;
			thrust::device_vector<int> neighborStartIndices;
			thrust::device_vector<int> adjList;
		};
	}  // namespace cuda
}  // namespace clustering









