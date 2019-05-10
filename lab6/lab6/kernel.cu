#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "helper_image.h"

#include <iostream>
#include <chrono>
#include <stack>

using namespace std;

struct rgb {
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

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
			outputMatrix[threadIdx.y + 2][threadIdx.x + 2] = matrix[yIndex + 1][xIndex + 1];
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

		const auto iMin = (yIndex == 0) ? 0 : -1;
		const auto iMax = (yIndex == height - 1) ? 0 : 1;
		const auto jMin = (xIndex == 0) ? 0 : -1;
		const auto jMax = (xIndex == width - 1) ? 0 : 1;

		for (auto i = iMin; i <= iMax; i++) {
			for (auto j = jMin; j <= jMax; j++) {
				if (minValue > outputMatrix[y + i][x + j]) {
					minValue = outputMatrix[y + i][x + j];
				}
			}
		}
		//__syncthreads();
		outputMatrix[y][x] = minValue;
		resultMatrix[yIndex][xIndex] = outputMatrix[y][x];
	}
}

__global__ void FilterGpuConv(rgb** matrix, rgb** resultMatrix, const int height, const int width) {

	__shared__ rgb outputMatrix[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

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
			outputMatrix[threadIdx.y + 2][threadIdx.x + 2] = matrix[yIndex + 1][xIndex + 1];
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

		const auto iMin = (y == 0) ? 0 : -1;
		const auto iMax = (y == height - 1) ? 0 : 1;
		const auto jMin = (x == 0) ? 0 : -1;
		const auto jMax = (x == width - 1) ? 0 : 1;

		rgb minValue = outputMatrix[y][x];
		auto minSum = outputMatrix[y][x].r + outputMatrix[y][x].g + outputMatrix[y][x].b;

		for (auto i = iMin; i <= iMax; i++) {
			for (auto j = jMin; j <= jMax; j++) {
				const auto newMinSum = outputMatrix[y + i][x + j].r + outputMatrix[y + i][x + j].g + outputMatrix[y + i][x + j].b;
				if (minSum > newMinSum && newMinSum != 0) {
					minSum = newMinSum;
					minValue = outputMatrix[y + i][x + j];
				}
			}
		}
		outputMatrix[y][x] = minValue;
		resultMatrix[yIndex][xIndex] = outputMatrix[y][x];
	}
}




class Image {
	const static unsigned int MAX_ASCII = 255;

	unsigned int height_;
	unsigned int width_;
	unsigned char* data_;
	rgb* convData_;
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


	rgb** initConvImageMatrixGpu() const {
		rgb** imageMatrix;
		cudaMallocManaged(&imageMatrix, width_ * 4 * height_ * sizeof(rgb*));
		for (auto i = 0; i < height_; i++) {
			cudaMallocManaged(&imageMatrix[i], width_ * 4 * sizeof(rgb));
			for (auto j = 0, k = 0; j < width_ * 4; j += 4, k++) {
				imageMatrix[i][k].r = data_[(i * width_ * 4) + j];
				imageMatrix[i][k].g = data_[(i * width_ * 4) + j + 1];
				imageMatrix[i][k].b = data_[(i * width_ * 4) + j + 2];
			}
		}
		return imageMatrix;
	}

	rgb** initConvImageMatrixCpu() const {
		rgb** imageMatrix = new rgb*[height_];
		//cudaMallocManaged(&imageMatrix, width_ * 4 * height_ * sizeof(rgb*));
		for (auto i = 0; i < height_; i++) {
			imageMatrix[i] = new rgb[width_ * 4];
			//cudaMallocManaged(&imageMatrix[i], width_ * 4 * sizeof(rgb));
			for (auto j = 0, k = 0; j < width_ * 4; j += 4, k++) {
				imageMatrix[i][k].r = data_[(i * width_ * 4) + j];
				imageMatrix[i][k].g = data_[(i * width_ * 4) + j + 1];
				imageMatrix[i][k].b = data_[(i * width_ * 4) + j + 2];
			}
		}
		return imageMatrix;
	}


	void writeImageMatrixToData(rgb** imageMatrix) const {
		for (auto i = 0; i < height_ ; i++) {
			for (auto j = 0, k = 0; j < width_ * 4; j += 4, k++) {
				data_[(i * width_ * 4) + j] = imageMatrix[i][k].r;
				data_[(i * width_ * 4) + j + 1] = imageMatrix[i][k].g;
				data_[(i * width_ * 4) + j + 2] = imageMatrix[i][k].b;
			}
		}
	}

	rgb getMinValueFromCell(rgb** imageMatrix, const int y, const int x) const {
		const auto iMin = (y == 0) ? 0 : -1;
		const auto iMax = (y == height_ - 1) ? 0 : 1;
		const auto jMin = (x == 0) ? 0 : -1;
		const auto jMax = (x == width_ - 1) ? 0 : 1;

		rgb minValue = imageMatrix[y][x];
		auto minSum = imageMatrix[y][x].r + imageMatrix[y][x].g + imageMatrix[y][x].b;

		for (auto i = iMin; i <= iMax; i++) {
			for (auto j = jMin; j <= jMax; j++) {
				const auto newMinSum = imageMatrix[y + i][x + j].r + imageMatrix[y + i][x + j].g + imageMatrix[y + i][x + j].b;
				//cout << (int)newMinSum << endl;
				if (minSum > newMinSum && newMinSum != 0) {
					minSum = newMinSum;
					minValue = imageMatrix[y + i][x + j];
				}
			}
		}
		return minValue;
	}

	void deleteData() const {
		delete data_;
	}

	void deleteImageMatrixCpu(rgb** imageMatrix) const {
		for (auto i = 0; i < height_; i++) {
			delete[] imageMatrix[i];
		}
		delete[] imageMatrix;
	}

	void deleteImageMatrixGpu(rgb** imageMatrix) const {
		for (auto i = 0; i < height_; i++) {
			cudaFree(&imageMatrix[i]);
		}
		cudaFree(&imageMatrix);
	}

	void showPixelsStdout(rgb** image) const {
		for (auto i = 0; i < height_ && i < 10; i++) {
			cout << endl;
			for (auto j = 0; j < width_ && j < 10; j+=4) {
				cout << static_cast<int>(image[i][j].r) << " "<< static_cast<int>(image[i][j].g) << " " << static_cast<int>(image[i][j].b) << "  ";
			}
		}
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

		if (!sdkLoadPPM4(filename.c_str(), &data_, &width_, &height_)) {
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
		if (!sdkSavePPM4ub(filename.c_str(), data_, width_, height_)) {
			cout << "Cannot save image file." << endl;
			throw std::invalid_argument("Cannot save image file.");
		}
	}

	void FilterImageGpu() const {
	
		rgb** imageMatrixConvGpu = initConvImageMatrixGpu();
		rgb** imageMatrixConvGpuResult = initConvImageMatrixGpu();

		float time = 0;
		cudaEvent_t start;
		cudaEvent_t stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		cudaEventRecord(start, nullptr);
		FilterGpuConv <<< numBlocks_, blockSize_ >>> (imageMatrixConvGpu, imageMatrixConvGpuResult, height_, width_);
		cudaEventRecord(stop, nullptr);
		cudaEventSynchronize(stop);

		cudaEventElapsedTime(&time, start, stop);
		cudaDeviceSynchronize();
		cout << endl << endl << "Time GPU: " << time << " ms. " << endl;
		cout << "Error: " << cudaGetErrorString(cudaGetLastError()) << endl;

		cudaEventDestroy(start);
		cudaEventDestroy(stop);

		cout << endl << "GPU RESULT: ";
		showPixelsStdout(imageMatrixConvGpuResult);
		writeImageMatrixToData(imageMatrixConvGpuResult);
		deleteImageMatrixGpu(imageMatrixConvGpu);
		deleteImageMatrixGpu(imageMatrixConvGpuResult);
	}

	void FilterImageCpu() const {
		const auto t1 = std::chrono::steady_clock::now();

		auto** imageMatrixCpu = initConvImageMatrixCpu();
		auto** imageMatrixCpuResult = initConvImageMatrixCpu();

		for (auto i = 0; i < height_; i++) {
			for (auto j = 0; j < width_; j++) {
				imageMatrixCpuResult[i][j] = getMinValueFromCell(imageMatrixCpu, i, j);
			}
		}

		const auto t2 = std::chrono::steady_clock::now();
		auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
		cout << "Time CPU: " << elapsedMs.count() << " ms" << endl;
		cout << "Error: " << cudaGetErrorString(cudaGetLastError()) << endl;
		cout << endl << "CPU : ";
		showPixelsStdout(imageMatrixCpu);
		cout << endl << endl << "CPU RESULT: ";
		showPixelsStdout(imageMatrixCpuResult);
		writeImageMatrixToData(imageMatrixCpuResult);
		deleteImageMatrixCpu(imageMatrixCpu);
		deleteImageMatrixCpu(imageMatrixCpuResult);
	}
};


const string FILE_NAME = "mojave.ppm";
const string NEW_FILE_NAME_CPU = "image_cpu.ppm";
const string NEW_FILE_NAME_GPU = "image_gpu.ppm";


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