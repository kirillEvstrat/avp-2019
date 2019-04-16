#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "helper_image.h"

#include <exception>
#include <iostream>
#include <chrono>
#include <stack>

using namespace std;

const int BLOCK_SIZE = 32;


//__device__ void GetMinValueFromCell(unsigned char imageMatrix[BLOCK_SIZE][BLOCK_SIZE],
//									const int height, const int width, const int y, const int x) {
//	const auto iMin = (y == 0) ? 0 : -1;
//	const auto iMax = (y == height - 1) ? 0 : 1;
//	const auto jMin = (x == 0) ? 0 : -1;
//	const auto jMax = (x == width - 1) ? 0 : 1;
//
//	char minValue = imageMatrix[y][x];
//
//	for (auto i = iMin; i <= iMax; i++) {
//		for (auto j = jMin; j <= jMax; j++) {
//			if (minValue > imageMatrix[y + i][x + j]) {
//				minValue = imageMatrix[y + i][x + j];
//			}
//		}
//	}
//
//	imageMatrix[y][x] = minValue;
//}

__global__ void FilterGpu(unsigned char** matrix, unsigned char** resultMatrix, const int height, const int width) {

	__shared__ unsigned char outputMatrix[BLOCK_SIZE][BLOCK_SIZE];

	const int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	const int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	if ((xIndex < width) && (yIndex < height)) {
		outputMatrix[threadIdx.y][threadIdx.x] = matrix[yIndex][xIndex];
		__syncthreads();

		auto x = threadIdx.x;
		auto y = threadIdx.y;

		char minValue = outputMatrix[y][x];

		minValue = ((y > 0) && (x > 0) && (minValue > outputMatrix[y - 1][x - 1])) ? outputMatrix[y - 1][x - 1] : minValue;
		minValue = ((y > 0) && (minValue > outputMatrix[y - 1][x])) ? outputMatrix[y - 1][x] : minValue;
		minValue = ((y > 0) && (x < (BLOCK_SIZE - 1)) && (minValue > outputMatrix[y - 1][x + 1])) ? outputMatrix[y - 1][x + 1] : minValue;
	
		minValue = ((x > 0) && (minValue > outputMatrix[y][x - 1])) ? outputMatrix[y][x - 1] : minValue;
		minValue = ((x < (BLOCK_SIZE - 1)) && (minValue > outputMatrix[y][x + 1])) ? outputMatrix[y][x + 1] : minValue;
		minValue = ((y < (BLOCK_SIZE - 1)) && (x > 0) && (minValue > outputMatrix[y + 1][x - 1])) ? outputMatrix[y + 1][x - 1] : minValue;
		minValue = ((y < (BLOCK_SIZE - 1)) && (minValue > outputMatrix[y + 1][x])) ? outputMatrix[y + 1][x] : minValue;
		minValue = ((y < (BLOCK_SIZE - 1)) && (x < (BLOCK_SIZE - 1)) && (minValue > outputMatrix[y + 1][x + 1])) ? outputMatrix[y + 1][x + 1] : minValue;
		__syncthreads();
		outputMatrix[y][x] = minValue;
		__syncthreads();
		resultMatrix[yIndex][xIndex] = outputMatrix[y][x]; // outputMatrix[threadIdx.x][threadIdx.y];
	}
}


class Image {
	const static unsigned int MAX_ASCII = 255;

	unsigned int height_;
	unsigned int width_;
	unsigned char* data_;
	unsigned char** imageMatrixCpu_;
	unsigned char** imageMatrixGpu_;
	dim3 blockSize_;
	dim3 numBlocks_;

public:

	explicit Image(const string filename) {
		LoadImage(filename);
	}

	~Image() {
		DeleteData();
		DeleteImageMatrix();
	}

	void LoadImage(const string filename) {
		data_ = nullptr;
		if (!sdkLoadPGM(filename.c_str(), &data_, &width_, &height_)) {
			cout << "Cannot read image file." << endl;
			throw std::invalid_argument("Cannot read image file.");
		}

		this->blockSize_ = dim3(BLOCK_SIZE, BLOCK_SIZE);
		this->numBlocks_ = dim3(height_ / BLOCK_SIZE + 1, width_ / BLOCK_SIZE + 1);
		
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

	unsigned char** InitImageMatrixCpu() const {
		auto** imageMatrix = new unsigned char*[height_];
		for (auto i = 0; i < height_; i++) {
			imageMatrix[i] = new unsigned char[width_];
			for (auto j = 0; j < width_; j++) {
				imageMatrix[i][j] = data_[(i * width_) + j];
			}
		}
		return imageMatrix;
	}

	unsigned char** InitImageMatrixGpu() const {
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


	void WriteImageMatrixToData(unsigned char** imageMatrix) const {
		cout << "\t\t" << static_cast<int>(data_[0]);
		for (auto i = 0; i < height_; i++) {
			cout << endl;
			for (auto j = 0; j < width_; j++) {
				data_[(i * width_) + j] = imageMatrix[i][j];
			}
		}
		cout << "\t\t" << static_cast<int>(data_[0]);
	}

	void FilterImageCpu() {
		imageMatrixCpu_ = InitImageMatrixCpu();
		unsigned char** imageMatrixCpuResult_ = InitImageMatrixCpu();
		cout << "\t\t" << static_cast<int>(imageMatrixCpu_[0][0]);
		for (auto i = 0; i < height_; i++) {
			for (auto j = 0; j < width_; j++) {
				imageMatrixCpuResult_[i][j] = GetMinValueFromCell(imageMatrixCpu_,i, j);
			}
		}
		cout << "\t\t" << static_cast<int>(imageMatrixCpuResult_[0][0]);
		WriteImageMatrixToData(imageMatrixCpuResult_);
	}

	void FilterImageGpu() {
		imageMatrixGpu_ = InitImageMatrixGpu(); 
		unsigned char** imageMatrixGpuNew = InitImageMatrixGpu();
		FilterGpu <<< numBlocks_, blockSize_ >>> (imageMatrixGpu_, imageMatrixGpuNew, height_, width_);
		cudaDeviceSynchronize();

		WriteImageMatrixToData(imageMatrixGpuNew);
	}

	char GetMinValueFromCell(unsigned char** imageMatrix, const int y, const int x) const {
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



	void DeleteData() const {
		delete data_;
	}

	void DeleteImageMatrix() const {
		for (auto i = 0; i < height_; i++) {
			delete[] imageMatrixCpu_[i];
		}
		delete[] imageMatrixCpu_;
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