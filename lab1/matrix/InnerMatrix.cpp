//
// Created by Rostislav Pekhovsky on 2019-02-07.
//

#include "InnerMatrix.h"


InnerMatrix::InnerMatrix(int size) : size(size) {

    this->pointer = new float *[size];
    for (int i = 0; i < size; i++) {
        this->pointer[i] = new float[size];
    }
}

void InnerMatrix::initRandomFloat() {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            this->pointer[i][j] = 300434.203f + i + j;

        }
    }
}

InnerMatrix *InnerMatrix::multiplyVectorized(InnerMatrix * __restrict__ matrix2) {

    auto *__restrict__ resMatrix = new InnerMatrix(this->size);
   // #pragma vector always
   for (int i = 0; i < size; i++) {
       auto * matrixVectorRes = resMatrix->pointer[i];
       for (int j = 0; j < size; j++) {
           auto *__restrict__ matrixVector1 = this->pointer[j];
           auto *__restrict__ matrixVector2 = matrix2->pointer[j];
           // #pragma vector always
           for (int k = 0; k < size; k++) {
               matrixVectorRes[k] += matrixVector1[k] * matrixVector2[k];
           }
       }
   }

    return resMatrix;
}


InnerMatrix *InnerMatrix::multiplyNotVectorized(InnerMatrix * matrix2) {

    auto *__restrict__ resMatrix = new InnerMatrix(this->size);

#pragma novector
    for (int i = 0; i < size; i++) {
        auto * matrixVectorRes = resMatrix->pointer[i];
        for (int j = 0; j < size; j++) {
            auto *__restrict__ matrixVector1 = this->pointer[j];
            auto *__restrict__ matrixVector2 = matrix2->pointer[j];

#pragma novector
            for (int k = 0; k < size; k++) {
                matrixVectorRes[k] += matrixVector1[k] * matrixVector2[k];
            }
        }
    }


    return resMatrix;
}
