#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "helper_image.h"

#include <iostream>
#include <chrono>
#include <stack>

using namespace std;


const int BLOCK_SIZE = 27;


__global__ void FilterGpu(unsigned char** matrix, unsigned char** resultMatrix, const int height, const int width) {

	__shared__ unsigned char outputMatrix[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	if ((xIndex < width) && (yIndex < height)) {

		outputMatrix[threadIdx.y + 1][threadIdx.x + 1] = matrix[yIndex][xIndex];

		if ((threadIdx.y == 0) && (threadIdx.x == 0) && (yIndex > 0) && (xIndex > 0)) {
			outputMatrix[threadIdx.y][threadIdx.x] = matrix[yIndex - 1][xIndex - 1];
		}
		if ((threadIdx.y == 0) && (threadIdx.x == BLOCK_SIZE - 1) && (yIndex > 0) && (xIndex < width - 1)) {
			outputMatrix[threadIdx.y][threadIdx.x + 2] = matrix[yIndex - 1][xIndex + 1];
		}
		if ((threadIdx.y == BLOCK_SIZE - 1) && (threadIdx.x == 0) && (yIndex < height - 1) && (xIndex > 0)) {
			outputMatrix[threadIdx.y + 2][threadIdx.x] = matrix[yIndex - 1][xIndex + 1];
		}
		if ((threadIdx.y == BLOCK_SIZE - 1) && (threadIdx.x == BLOCK_SIZE - 1) && (yIndex < height - 1) && (xIndex < width - 1)) {
			outputMatrix[threadIdx.y + 2][threadIdx.x + 2] = matrix[yIndex +  1][xIndex + 1];
		}
		if ((threadIdx.y == 0) && (yIndex > 0)) {
			outputMatrix[threadIdx.y][threadIdx.x + 1] = matrix[yIndex - 1][xIndex];
		}
		if (threadIdx.x == 0 && xIndex > 0) {
			outputMatrix[threadIdx.y + 1][threadIdx.x] = matrix[yIndex][xIndex - 1];
		}
		if (threadIdx.y == BLOCK_SIZE - 1 && yIndex < height - 1) {
			outputMatrix[threadIdx.y + 2][threadIdx.x + 1] = matrix[yIndex + 1][xIndex];
		}
		if (threadIdx.x == BLOCK_SIZE - 1 && xIndex < width - 1) {
			outputMatrix[threadIdx.y + 1][threadIdx.x + 2] = matrix[yIndex][xIndex + 1];
		}
		__syncthreads();

		const auto x = threadIdx.x + 1;
		const auto y = threadIdx.y + 1;

		char minValue = outputMatrix[y][x];
		
		const auto iMin = (y == 0) ? 0 : -1;
		const auto iMax = (y == height - 1) ? 0 : 1;
		const auto jMin = (x == 0) ? 0 : -1;
		const auto jMax = (x == width - 1) ? 0 : 1;

		for (auto i = iMin; i <= iMax; i++) {
			for (auto j = jMin; j <= jMax; j++) {
				if (minValue > outputMatrix[y + i][x + j]) {
					minValue = outputMatrix[y + i][x + j];
				}
			}
		}
		__syncthreads();

		outputMatrix[y][x] = minValue;
		__syncthreads();
		
		resultMatrix[yIndex][xIndex] = outputMatrix[y][x];
	}
}


class Image {
	const static unsigned int MAX_ASCII = 255;

	unsigned int height_;
	unsigned int width_;
	unsigned char* data_;
	dim3 blockSize_;
	dim3 numBlocks_;

	unsigned char** initImageMatrixCpu() const {
		auto** imageMatrix = new unsigned char*[height_];
		for (auto i = 0; i < height_; i++) {
			imageMatrix[i] = new unsigned char[width_];
			for (auto j = 0; j < width_; j++) {
				imageMatrix[i][j] = data_[(i * width_) + j];
			}
		}
		return imageMatrix;
	}

	unsigned char** initImageMatrixGpu() const {
		unsigned char** imageMatrix;
		cudaMallocManaged(&imageMatrix, width_ * height_ * sizeof(char*));
		for (auto i = 0; i < height_; i++) {
			cudaMallocManaged(&imageMatrix[i], width_ * sizeof(char));
			for (auto j = 0; j < width_; j++) {
				imageMatrix[i][j] = data_[(i * width_) + j];
			}
		}
		return imageMatrix;
	}

	void writeImageMatrixToData(unsigned char** imageMatrix) const {
		cout << "\t\t" << static_cast<int>(data_[0]);
		for (auto i = 0; i < height_; i++) {
			for (auto j = 0; j < width_; j++) {
				data_[(i * width_) + j] = imageMatrix[i][j];
			}
		}
		cout << "\t\t" << static_cast<int>(data_[0]);
	}

	char getMinValueFromCell(unsigned char** imageMatrix, const int y, const int x) const {
		const auto iMin = (y == 0) ? 0 : -1;
		const auto iMax = (y == height_ - 1) ? 0 : 1;
		const auto jMin = (x == 0) ? 0 : -1;
		const auto jMax = (x == width_ - 1) ? 0 : 1;

		char minValue = imageMatrix[y][x];

		for (auto i = iMin; i <= iMax; i++) {
			for (auto j = jMin; j <= jMax; j++) {
				if (minValue > imageMatrix[y + i][x + j]) {
					minValue = imageMatrix[y + i][x + j];
				}
			}
		}
		return minValue;
	}

	void deleteData() const {
		delete data_;
	}

	void deleteImageMatrixCpu(unsigned char** imageMatrix) const {
		for (auto i = 0; i < height_; i++) {
			delete[] imageMatrix[i];
		}
		delete[] imageMatrix;
	}

	void deleteImageMatrixGpu(unsigned char** imageMatrix) const {
		for (auto i = 0; i < height_; i++) {
			cudaFree(&imageMatrix[i]);
		}
		cudaFree(&imageMatrix);
	}


public:

	explicit Image(const string filename) {
		LoadImage(filename);
	}

	~Image() {
		deleteData();
	}

	void LoadImage(const string filename) {
		data_ = nullptr;
		if (!sdkLoadPGM(filename.c_str(), &data_, &width_, &height_)) {
			cout << "Cannot read image file." << endl;
			throw std::invalid_argument("Cannot read image file.");
		}

		this->blockSize_ = dim3(BLOCK_SIZE, BLOCK_SIZE);
		this->numBlocks_ = dim3(width_ / BLOCK_SIZE + 1, height_ / BLOCK_SIZE + 1);
		
		cout << "Block size, x: " << blockSize_.x << ", y: " << blockSize_.y << endl;
		cout << "Num of blocks, x: " << numBlocks_.x << ", y: " << numBlocks_.y << endl;
		cout << "Width: " << width_ << endl;
		cout << "Height: " << height_ << endl;
	}

	void SaveImage(const string filename) const {
		if (!sdkSavePGM(filename.c_str(), data_, width_, height_)) {
			cout << "Cannot save image file." << endl;
			throw std::invalid_argument("Cannot save image file.");
		}
	}

	void FilterImageGpu() const {
		auto** imageMatrixGpu = initImageMatrixGpu();
		auto** imageMatrixGpuResult = initImageMatrixGpu();
		
		float time = 0;
		cudaEvent_t start;
		cudaEvent_t stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start, nullptr);
		FilterGpu <<< numBlocks_, blockSize_ >>> (imageMatrixGpu, imageMatrixGpuResult, height_, width_);
		cudaEventRecord(stop, nullptr);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&time, start, stop);
		cudaDeviceSynchronize();
		cout << "Time GPU: " << time << " ms. " << endl;
		cout << "Error: " << cudaGetErrorString(cudaGetLastError()) << endl;

		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		
		cudaDeviceSynchronize();
		cout << endl << "Error: " << cudaGetErrorString(cudaGetLastError()) << endl;
		writeImageMatrixToData(imageMatrixGpuResult);
		
		deleteImageMatrixGpu(imageMatrixGpu);
		deleteImageMatrixGpu(imageMatrixGpuResult);
	}

	void FilterImageCpu() const {
		const auto t1 = std::chrono::steady_clock::now();
		
		auto** imageMatrixCpu = initImageMatrixCpu();
		auto** imageMatrixCpuResult = initImageMatrixCpu();

		for (auto i = 0; i < height_; i++) {
			for (auto j = 0; j < width_; j++) {
				imageMatrixCpuResult[i][j] = getMinValueFromCell(imageMatrixCpu, i, j);
			}
		}

		const auto t2 = std::chrono::steady_clock::now();
		auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
		cout << "Time CPU: " << elapsedMs.count() << " ms" << endl;
		cout << "Error: " << cudaGetErrorString(cudaGetLastError()) << endl;

		writeImageMatrixToData(imageMatrixCpuResult);
		deleteImageMatrixCpu(imageMatrixCpu);
		deleteImageMatrixCpu(imageMatrixCpuResult);
	}
};


const string FILE_NAME = "image3.pgm";
const string NEW_FILE_NAME_CPU = "image_cpu.pgm";
const string NEW_FILE_NAME_GPU = "image_gpu.pgm";


int main(int argv, char* args[]) {
	auto* imageCpu = new Image(FILE_NAME);
	auto* imageGpu = new Image(FILE_NAME);

	imageCpu->FilterImageCpu();
	imageCpu->SaveImage(NEW_FILE_NAME_CPU);

	imageGpu->FilterImageGpu();
	imageGpu->SaveImage(NEW_FILE_NAME_GPU);
	
	delete imageCpu;
	delete imageGpu;
	return 0;
}