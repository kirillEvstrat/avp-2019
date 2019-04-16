#include "device_launch_parameters.h"
#include "cuda_runtime.h"

#include <ctime>
#include <iostream>
#include <chrono>

using namespace std;

const int BLOCK_SIZE = 32;
const int MATRIX_WIDTH = 1233;
const int MATRIX_HEIGHT = 5433;

/*
 * __global__ void RotateMatrixGpu(short* pointerOld[], short* pointerNew[], const int width, const int height) {

	__shared__ short outputMatrix[BLOCK_SIZE][BLOCK_SIZE];
	
	__shared__ int outputMatrixInt[BLOCK_SIZE][BLOCK_SIZE];

	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	if ((xIndex < height) && (yIndex < width)) {
		outputMatrix[threadIdx.x][threadIdx.y] = pointerOld[height - xIndex - 1][width - yIndex - 1];
		__syncthreads();
		pointerNew[xIndex][yIndex] = outputMatrix[threadIdx.x][threadIdx.y];
	}	
}
 */


__global__ void RotateMatrixGpu(void** pointerOld, void** pointerNew, const int width, const int height) {

	__shared__ int outputMatrixInt[BLOCK_SIZE][BLOCK_SIZE];

	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
	
	const auto isOdd = width % 2 != 0;
	
	auto widthInt = width / 2;
	if (isOdd) {
		widthInt++;
	}

	auto** pointerOldInt = reinterpret_cast<int**>(pointerOld);
	auto** pointerNewInt = reinterpret_cast<int**>(pointerNew);

	if ((xIndex < height) && (yIndex < widthInt)) {
		outputMatrixInt[threadIdx.x][threadIdx.y] = pointerOldInt[height - xIndex - 1][widthInt - yIndex - 1];
		outputMatrixInt[threadIdx.x][threadIdx.y] = outputMatrixInt[threadIdx.x][threadIdx.y] << 16 | outputMatrixInt[threadIdx.x][threadIdx.y] >> 16;
		__syncthreads();

		if (isOdd) {
			*(reinterpret_cast<short*>(pointerNewInt[xIndex] + yIndex)) = outputMatrixInt[threadIdx.x][threadIdx.y] >> 16;
			if (yIndex > 0) {
				*(reinterpret_cast<short*>(pointerNewInt[xIndex] + yIndex) - 1) = outputMatrixInt[threadIdx.x][threadIdx.y];
			}

		} else {
			pointerNewInt[xIndex][yIndex] = outputMatrixInt[threadIdx.x][threadIdx.y];
		}
	}
}

class Matrix {

	const static int MAX_STDOUT_SIZE = 10;

	short** pointerCpu_;
	short** pointerCpuResult_;
	short** pointerGpu_;
	short** pointerGpuResult_;

	int width_;
	int height_;

	dim3 blockSize_;
	dim3 numBlocks_;

	void AllocateMemory() {
		pointerCpu_ = AllocateCpuMatrix(width_, height_);
		pointerCpuResult_ = AllocateCpuMatrix(width_, height_);
		pointerGpu_ = AllocateGpuMatrix(width_, height_);
		pointerGpuResult_ = AllocateGpuMatrix(width_, height_);
	}

	void InitValues() const {
		auto counter = 0;
		for (auto i = 0; i < height_; i++) {
			for (auto j = 0; j < width_; j++) {
				counter++;
				if (counter > 10000) {
					counter = 0;
				}
				pointerCpu_[i][j] = counter;
				pointerGpu_[i][j] = counter;
				pointerCpuResult_[i][j] = 0;
				pointerGpuResult_[i][j] = 0;
			}
		}
	}

public:
	explicit Matrix(const  int width, const int height) {
		this->width_ = width;
		this->height_ = height;
		this->blockSize_ = dim3(BLOCK_SIZE, BLOCK_SIZE);
		this->numBlocks_ = dim3(height_ / BLOCK_SIZE + 1, width_ / 2 / BLOCK_SIZE + 1);

		cout << "Matrix width, x: " << width_ << endl;
		cout << "Matrix height, x: " << height_ << endl;
		cout << "Block size, x: " << blockSize_.x << ", y: " << blockSize_.y << endl;
		cout << "Num of blocks, x: " << numBlocks_.x << ", y: " << numBlocks_.y << endl;
		AllocateMemory();
		InitValues();
	}

	~Matrix() {
		DeleteCpuMatrix(pointerCpu_, height_);
		DeleteCpuMatrix(pointerCpuResult_, height_);
		DeleteGpuMatrix(pointerGpu_, height_);
		DeleteGpuMatrix(pointerGpuResult_, height_);
	}

	static short** AllocateCpuMatrix(const int width, const int height) {
		auto** pointer = new short*[height];
		for (auto i = 0; i < height; i++) {
			pointer[i] = new short[width];
		}
		return pointer;
	}

	static void DeleteCpuMatrix(short** pointer, const int height) {
		for (auto i = 0; i < height; i++) {
			delete[] pointer[i];
		}
		delete[] pointer;
	}

	static short** AllocateGpuMatrix(const int width, const int height) {
		short** pointer;
		cudaMallocManaged(&pointer, width * height * sizeof(short));
		for (auto i = 0; i < height; i++) {
			cudaMallocManaged(&pointer[i], width * sizeof(short));
		}
		return pointer;
	}

	static void DeleteGpuMatrix(short** pointer, const int height) {
		for (auto i = 0; i < height; i++) {
			cudaFree(&pointer[i]);
		}
		cudaFree(&pointer);
	}

	void RotateCpu() const {
		for (auto i = 0; i < height_; i++) {
			for (auto j = 0; j < width_; j++) {
				pointerCpuResult_[i][j] = pointerCpu_[height_ - i - 1][width_ - j - 1];
			}
		}
	}

	void RotateGpu() const {
		RotateMatrixGpu << < numBlocks_, blockSize_ >> > ((void**)pointerGpu_, (void**)pointerGpuResult_, width_, height_);
	}

	static void ShowMatrix(short** pointer, const int width, const int height) {
		for (auto i = 0; i < height && i < MAX_STDOUT_SIZE; i++) {
			cout << endl;
			for (auto j = 0; j < width && j < MAX_STDOUT_SIZE; j++) {
				cout << pointer[i][j] << "\t";
			}
		}
		cout << endl;
		if (width > MAX_STDOUT_SIZE && height > MAX_STDOUT_SIZE) {
			cout << endl;
			for (auto i = height - 10; i < height; i++) {
				cout << endl;
				for (auto j = width - 10; j < width; j++) {
					cout << pointer[i][j] << "\t";
				}
			}
		}
		cout << endl;
	}

	void ShowStdout() const {
		std::cout << "\n\nCPU: " << std::endl;
		ShowMatrix(pointerCpu_, width_, height_);
		std::cout << "\n\nGPU: " << std::endl;
		ShowMatrix(pointerGpu_, width_, height_);
	}

	void ShowResultStdout() const {
		std::cout << "\n\nCPU RESULT: " << std::endl;
		ShowMatrix(pointerCpuResult_, width_, height_);
		std::cout << "\n\nGPU RESULT: " << std::endl;
		ShowMatrix(pointerGpuResult_, width_, height_);
	}
};


void MeasurePerformCpu(Matrix* matrix) {
	const auto t1 = std::chrono::steady_clock::now();
	matrix->RotateCpu();
	const auto t2 = std::chrono::steady_clock::now();

	auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
	cout << "Time CPU: " << elapsedMs.count() << " ms" << endl;
	cout << "Error: " << cudaGetErrorString(cudaGetLastError()) << endl;

}

void MeasurePerformGpu(Matrix* matrix) {
	float time = 0;
	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, nullptr);
	matrix->RotateGpu();
	cudaEventRecord(stop, nullptr);
	cudaEventSynchronize(stop);

	cudaEventElapsedTime(&time, start, stop);
	cudaDeviceSynchronize();
	cout << "Time GPU: " << time << " ms. " << endl;
	cout << "Error: " << cudaGetErrorString(cudaGetLastError()) << endl;

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}


int main(int argv, char* args[]) {

	auto* matrix = new Matrix(MATRIX_WIDTH, MATRIX_HEIGHT);

	MeasurePerformCpu(matrix);
	MeasurePerformGpu(matrix);
	matrix->ShowStdout();
	matrix->ShowResultStdout();

	delete matrix;
	return 0;
}