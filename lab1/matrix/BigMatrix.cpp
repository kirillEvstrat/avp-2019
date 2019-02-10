//
// Created by Rostislav Pekhovsky on 2019-02-07.
//

#include "BigMatrix.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"


BigMatrix::BigMatrix(int size, int innerMatrixSize) {
    this->size = size;
    this->pointer = new InnerMatrix**[size];
    for (int i = 0; i < size; i++) {
        this->pointer[i] = new InnerMatrix*[size];
        for (int j = 0; j < size; j++) {
            this->pointer[i][j] = new InnerMatrix(innerMatrixSize);
        }
    }
}


void BigMatrix::InitRandomFloat() {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            pointer[i][j]->InitRandomFloat();
        }
    }
}


BigMatrix *BigMatrix::MultiplyVectorized(BigMatrix* matrix) {

    auto* resultMatrix = new BigMatrix(size, 0);


    for (int i = 0; i < size; i++) {
        auto* __restrict__ matrix1Column = this->pointer[i];
        auto* __restrict__ resultMatrixRow = resultMatrix->pointer[i];

        for (int j = 0; j < size; j++) {
            auto* __restrict__ matrix2Column = matrix->pointer[j];

            for (int k = 0; k < size; k++) {
                resultMatrixRow[k]->AddVectorized(matrix1Column[j]->MultiplyVectorized(matrix2Column[k]));
            }
        }
    }
    return resultMatrix;
}



BigMatrix *BigMatrix::MultiplyNotVectorized(BigMatrix *matrix) {

    auto* resultMatrix = new BigMatrix(size, 0);

    #pragma novector
    for (int i = 0; i < size; i++) {
        auto* __restrict__ matrix1Column = this->pointer[i];
        auto* __restrict__ resultMatrixRow = resultMatrix->pointer[i];

        for (int j = 0; j < size; j++) {
            auto* __restrict__ matrix2Column = matrix->pointer[j];


            for (int k = 0; k < size; k++) {
                resultMatrixRow[k]->AddNotVectorized(matrix1Column[j]->MultiplyNotVectorized(matrix2Column[k]));
            }
        }
    }
    return resultMatrix;
}


#pragma clang diagnostic pop