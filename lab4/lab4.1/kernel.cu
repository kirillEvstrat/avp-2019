#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <ctime>
#include <iostream>
#include <chrono>

#define MATRIX_SIZE 4096
#define BLOCK_SIZE 32 //MAX - 32

using namespace std;


__global__ void rotate_matrix_gpu(short* pointerOld[], short* pointerNew[]) {
	__shared__ short output_matrix[BLOCK_SIZE][BLOCK_SIZE];
	
	const int x_index = blockIdx.x * blockDim.x + threadIdx.x;
	const int y_index = blockIdx.y * blockDim.y + threadIdx.y;

	if ((x_index < MATRIX_SIZE) && (y_index < MATRIX_SIZE)) {
		output_matrix[threadIdx.x][threadIdx.y] = pointerOld[MATRIX_SIZE - x_index - 1][MATRIX_SIZE - y_index - 1];
		__syncthreads();
		pointerNew[x_index][y_index] = output_matrix[threadIdx.x][threadIdx.y];
	} else {
		return;
	}
}

class Matrix {
	short** pointer_;
	short** pointerGpu_;
	short** pointerGpuResult_;
	int size_;
	dim3 blockSize_;
	dim3 numBlocks_;
	const static int maxStdoutSize_ = 10;

public:
	explicit Matrix(const int size) {
		this->size_ = size;	
		this->blockSize_ = dim3(BLOCK_SIZE, BLOCK_SIZE);
		this->numBlocks_ = dim3(size_ / BLOCK_SIZE, size_ / BLOCK_SIZE);

		cout << "Matrix size, x: " << size_ << endl;
		cout << "Block size, x: " << blockSize_.x << ", y: "<< blockSize_.y << endl;
		cout << "Num of blocks, x: " << numBlocks_.x << ", y: " << numBlocks_.y << endl;

		this->pointer_ = new short*[size];
		for (auto i = 0; i < size_; i++) {
			this->pointer_[i] = new short[size];
		}

		cudaMallocManaged(&pointerGpu_, size_* size_ *sizeof(short));
		for (auto i = 0; i < size_; i++) {
			cudaMallocManaged(&pointerGpu_[i], size_ * sizeof(short));
		}
		cudaMallocManaged(&pointerGpuResult_, size_* size_ * sizeof(short));
		for (auto i = 0; i < size_; i++) {
			cudaMallocManaged(&pointerGpuResult_[i], size_ * sizeof(short));
		}
	
		InitValues();
	}

	void InitValues() const {
		auto counter = 0;
		for (auto i = 0; i < size_; i++) {
			for (auto j = 0; j < size_; j++) {
				counter++;
				if (counter > 10000)
				{
					counter = 0;
				}
				pointer_[i][j] = counter;
				pointerGpu_[i][j] = counter;
				pointerGpuResult_[i][j] = 0;
			}
		}
	}

	void RotateGpu() const {
		rotate_matrix_gpu<<<numBlocks_, blockSize_>>>(pointerGpu_, pointerGpuResult_);
		//cudaDeviceSynchronize();
	}

	void RotateCpu() const {
		auto counter = 0;
		const auto max_count = size_ * size_ / 2;
		for (auto i = 0; i < size_; i++) {
			for (auto j = 0; j < size_; j++) {
				if (max_count <= counter) {
					return;
				} 
				counter++;
				const auto temp = pointer_[i][j];
				pointer_[i][j] = pointer_[size_ - i - 1][size_ - j - 1];
				pointer_[size_ - i - 1][size_ - j - 1] = temp;
			}
		}
	}

	void ShowStdout() const {
		cout << endl << endl;
		cout << "CPU: ";
		for (auto i = 0; i < size_ && i < maxStdoutSize_; i++) {
			cout << endl;
			for (auto j = 0; j < size_ && j < maxStdoutSize_; j++) {
				cout << pointer_[i][j] << "\t";
			}
		}
		cout << endl;
		for (auto i = size_ - 10; i < size_; i++) {
			cout << endl;
			for (auto j = size_ - 10; j < size_; j++) {
				cout << pointer_[i][j] << "\t";
			}
		}
		cout << endl << endl;
		cout << "GPU: ";
		for (auto i = 0; i < size_ && i < maxStdoutSize_; i++) {
			cout << endl;
			for (auto j = 0; j < size_ && j < maxStdoutSize_; j++) {
				cout << pointerGpu_[i][j] << "\t";
			}
		}
		cout << endl;
		for (auto i = size_ - 10; i < size_; i++) {
			cout << endl;
			for (auto j = size_ - 10; j < size_; j++) {
				cout << pointerGpu_[i][j] << "\t";
			}
		}
		cout << endl << endl;
		cout << "GPU Result: ";
		for (auto i = 0; i < size_ && i < maxStdoutSize_; i++) {
			cout << endl;
			for (auto j = 0; j < size_  && j < maxStdoutSize_; j++) {
				cout << pointerGpuResult_[i][j] << "\t";
			}
		}		
		cout << endl;
		for (auto i = size_ - 10; i < size_; i++) {
			cout << endl;
			for (auto j = size_ - 10; j < size_; j++) {
				cout << pointerGpuResult_[i][j] << "\t";
			}
		}
		

	}

	~Matrix() {
		for (auto i = 0; i < size_; i++) {
			cudaFree(&pointerGpu_[i]);
		}
		cudaFree(&pointerGpuResult_);
		
		for (auto i = 0; i < size_; i++) {
			cudaFree(&pointerGpuResult_[i]);
		}
		cudaFree(&pointerGpuResult_);
		
		for (auto i = 0; i < size_; i++) {
			delete[] pointer_[i];
		}
		delete[] pointer_;
	}
};

const int matrix_width = 10;
const int matrix_height = 10;


void measure_perform_cpu(Matrix* matrix) {

	const auto t1 = std::chrono::steady_clock::now();
	matrix->RotateCpu();
	const auto t2 = std::chrono::steady_clock::now();
	
	auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
	cout << "Time CPU: " << elapsed_ms.count() << " ms" << endl;
	cout << "Error: " << cudaGetErrorString(cudaGetLastError()) << endl;

}


void measure_perform_gpu(Matrix* matrix) {
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
	srand(time(nullptr));
	auto* matrix = new Matrix(MATRIX_SIZE);
	
	measure_perform_cpu(matrix);
	measure_perform_gpu(matrix);

	matrix->ShowStdout();

	delete matrix;
    return 0;
}

