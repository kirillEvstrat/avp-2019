//
// Created by Rostislav Pekhovsky on 2019-02-07.
//


#include <xmmintrin.h>

#include "InnerMatrix.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"

InnerMatrix::InnerMatrix(int size) : size(size) {

    this->pointer = new float* [size];
    for (int i = 0; i < size; i++) {
        this->pointer[i] = new float[size];
    }
}

void InnerMatrix::InitRandomFloat() {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            this->pointer[i][j] = i + j + 5;
        }
    }
}

InnerMatrix* InnerMatrix::MultiplyVectorized(const InnerMatrix* __restrict__ matrix) {

    auto* resultMatrix = new InnerMatrix(size);

#pragma vector always
    for (int i = 0; i < size; i++) {
        auto* __restrict__ matrix1Column = this->pointer[i];
        auto* __restrict__ resultMatrixRow = resultMatrix->pointer[i];

        for (int j = 0; j < size; j++) {
            auto* __restrict__ matrix2Column = matrix->pointer[j];

#pragma vector always
            for (int k = 0; k < size; k++) {
                resultMatrixRow[k] += matrix1Column[j] * matrix2Column[k];
            }
        }
    }
    return resultMatrix;
}


InnerMatrix* InnerMatrix::MultiplyNotVectorized(const InnerMatrix* matrix) {

    auto* resultMatrix = new InnerMatrix(size);

#pragma novector
    for (int i = 0; i < size; i++) {
        auto* matrix1Column = this->pointer[i];
        auto* resultMatrixRow = resultMatrix->pointer[i];

        for (int j = 0; j < size; j++) {
            auto* matrix2Column = matrix->pointer[j];

#pragma novector
            for (int k = 0; k < size; k++) {
                resultMatrixRow[k] += matrix1Column[j] * matrix2Column[k];

//                std::cout << "I: " << i << ", J: " << j << ", K: " << k << std::endl;
//                std::cout << "Matrix 1 col: ";
//                for (int n = 0; n < size; n++) {
//                    std::cout << this->pointer[i][n] << " ";
//                }
//                std::cout << std::endl;
//                std::cout << "Matrix 2 col: ";
//                for (int n = 0; n < size; n++) {
//                    std::cout << matrix->pointer[j][n] << " ";
//                }
//                std::cout << std::endl;
//                std::cout << "Res matr row: ";
//                for (int n = 0; n < size; n++) {
//                    std::cout << resultMatrix->pointer[i][n] << " ";
//
//                }
            }
        }
    }

    return resultMatrix;
}


InnerMatrix* InnerMatrix::MultiplyManuallyVectorized(const InnerMatrix* matrix) {

    auto* resultMatrix = new InnerMatrix(size);
    __m128 matrix1ColumnSSE;
    __m128 matrix2ColumnSSE;
    __m128 resultMatrixRowSSE;

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {

            __m128 valueFromMatrixA = _mm_set1_ps(this->pointer[i][j]);

            for (int k = 0; k < size; k += 4) {
                matrix2ColumnSSE = _mm_load_ps(matrix->pointer[j] + k);
                resultMatrixRowSSE = _mm_load_ps(resultMatrix->pointer[i] + k);
                resultMatrixRowSSE = _mm_add_ps(resultMatrixRowSSE, _mm_mul_ps(valueFromMatrixA, matrix2ColumnSSE));

                _mm_store_ps(resultMatrix->pointer[i] + k, resultMatrixRowSSE);
               // std::cout << std::endl;
            }
        }
    }
    return resultMatrix;
}
//std::cout << "I: " << i << ", J: " << j << ", K: " << k << ", L: " << l << std::endl;
//    std::cout << "Matrix 1 col: ";
//                for (int n = 0; n < size; n++) {
//                    std::cout << this->pointer[l][n] << " ";
//                }
//                std::cout << std::endl;
//                std::cout << "Matrix 2 col: ";
//                for (int n = 0; n < size; n++) {
//                    std::cout << matrix->pointer[l][n] << " ";
//                }
//                std::cout << std::endl;
//                std::cout << "Res matr row: ";
//                for (int n = 0; n < size; n++) {
//                    std::cout << resultMatrix->pointer[i][n] << " ";
//                }
//                std::cout << std::endl;

//    for (int i = 0; i < size; i++) {
//
//        for (int j = 0; j < size; j++) {
//
//            for (int k = 0; k < size; k++) {
//                resultMatrix->pointer[i][j] += this->pointer[i][k] * matrix->pointer[k][j];
//                std::cout << "Add: " << resultMatrix->pointer[i][j] << std::endl;
//                std::cout << "I: " << i << ", J: "<< j << ", K: " << k << std::endl;
//            }
//
//        }
//    }

void InnerMatrix::AddVectorized(const InnerMatrix* matrix) {

#pragma vector always
    for (int i = 0; i < size; i++) {
        auto* matrix1Row = this->pointer[i];
        auto* matrix2Row = matrix->pointer[i];

#pragma vector always
        for (int j = 0; j < size; j++) {
            matrix1Row[j] += matrix2Row[j];
        }
    }
}

void InnerMatrix::AddNotVectorized(const InnerMatrix* matrix) {

#pragma novector
    for (int i = 0; i < size; i++) {
        auto* matrix1Row = this->pointer[i];
        auto* matrix2Row = matrix->pointer[i];

#pragma novector
        for (int j = 0; j < size; j++) {
            matrix1Row[j] += matrix2Row[j];
        }
    }
}

void InnerMatrix::ShowStdout() {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << " [" << this->pointer[i][j] << "]";
        }
        std::cout << std::endl;
    }
}

float** InnerMatrix::GetPointer() const {
    return pointer;
}


#pragma clang diagnostic pop