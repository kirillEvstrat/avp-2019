//
// Created by Rostislav Pekhovsky on 2019-02-07.
//

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
            this->pointer[i][j] = RandomFloat();
        }
    }
}


InnerMatrix* InnerMatrix::MultiplyVectorized(InnerMatrix* __restrict__ matrix) {

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


InnerMatrix* InnerMatrix::MultiplyNotVectorized(InnerMatrix* matrix) {

    auto* resultMatrix = new InnerMatrix(size);

    #pragma novector
    for (int i = 0; i < size; i++) {
        auto* __restrict__ matrix1Column = this->pointer[i];
        auto* __restrict__ resultMatrixRow = resultMatrix->pointer[i];

        for (int j = 0; j < size; j++) {
            auto* __restrict__ matrix2Column = matrix->pointer[j];

            #pragma novector
            for (int k = 0; k < size; k++) {
                resultMatrixRow[k] += matrix1Column[j] * matrix2Column[k];
            }
        }
    }
    return resultMatrix;
}

InnerMatrix* InnerMatrix::Multiply(InnerMatrix* matrix) {

    auto* resultMatrix = new InnerMatrix(size);

    for (int i = 0; i < size; i++) {
        auto* __restrict__ matrix1Column = this->pointer[i];
        auto* __restrict__ resultMatrixRow = resultMatrix->pointer[i];

        for (int j = 0; j < size; j++) {
            auto* __restrict__ matrix2Column = matrix->pointer[j];

            for (int k = 0; k < size; k++) {
                resultMatrixRow[k] += matrix1Column[j] * matrix2Column[k];
            }
        }
    }
    return resultMatrix;
}

void InnerMatrix::AddVectorized(InnerMatrix* matrix) {

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

void InnerMatrix::AddNotVectorized(InnerMatrix* matrix) {

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
            std::cout << " " << this->pointer[i][j];
        }
        std::cout << std::endl;
    }
}

float** InnerMatrix::GetPointer() const {
    return pointer;
}


#pragma clang diagnostic pop