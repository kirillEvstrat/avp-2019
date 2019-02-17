//
// Created by Rostislav Pekhovsky on 2019-02-07.
//

#include "BigMatrix.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-pragmas"


BigMatrix::BigMatrix(int size, int innerMatrixSize) {
    this->size = size;
    this->innerMatrixSize = innerMatrixSize;
    this->pointer = new InnerMatrix**[size];
    for (int i = 0; i < size; i++) {
        this->pointer[i] = new InnerMatrix* [size];
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


BigMatrix* BigMatrix::MultiplyVectorized(const BigMatrix* __restrict__ matrix) {

    auto* resultMatrix = new BigMatrix(size, innerMatrixSize);

    const auto size = this->size;

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


BigMatrix* BigMatrix::MultiplyNotVectorized(const BigMatrix* __restrict__ matrix) {

    auto* __restrict__ resultMatrix = new BigMatrix(size, innerMatrixSize);
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


BigMatrix* BigMatrix::MultiplyManuallyVectorized(const BigMatrix* __restrict__ matrix) {

    const auto size = this->size;

    auto* __restrict__ resultMatrix = new BigMatrix(size, innerMatrixSize);

    for (int i = 0; i < size; i++) {
        auto* __restrict__ matrix1Column = this->pointer[i];
        auto* __restrict__ resultMatrixRow = resultMatrix->pointer[i];

        for (int j = 0; j < size; j++) {
            auto* __restrict__ matrix2Column = matrix->pointer[j];


            for (int k = 0; k < size; k++) {
                resultMatrix->pointer[i][k]->AddVectorized(this->pointer[i][j]->MultiplyManuallyVectorized(matrix->pointer[j][k]));
            }
        }
    }
    return resultMatrix;
}


void BigMatrix::ShowStdout() {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            std::cout << "Matrix " << i << " " << j << std::endl;
            this->pointer[i][j]->ShowStdout();
        }
    }
}

bool BigMatrix::operator==(const BigMatrix &matrix) const {
    if (size == matrix.size && innerMatrixSize == matrix.innerMatrixSize) {
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                if (*this->pointer[i][j] != *matrix.pointer[i][j]) {
                    std::cout << "BIG I: " << i << "J: " << j << std::endl;
                    return false;
                }
            }
        }
        return true;
    } else {
        std::cout << "ddd";
        return false;
    }
}

bool BigMatrix::operator!=(const BigMatrix &rhs) const {
    return !(rhs == *this);
}

#pragma clang diagnostic pop