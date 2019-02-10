//
// Created by Rostislav Pekhovsky on 2019-02-07.
//

#include "OuterMatrix.h"

OuterMatrix::OuterMatrix(int size) {
    this->size = size;
    this->pointer = new InnerMatrix**[size];
    for (int i = 0; i < size; i++) {
        this->pointer[i] = new InnerMatrix*[size];
        for (int j = 0; j < size; j++) {
            this->pointer[i][j] = new InnerMatrix(100);
        }
    }
}


OuterMatrix *OuterMatrix::multiplyVectorized(OuterMatrix *matrix2) {

    auto* result = new OuterMatrix(size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result->pointer[i][j] = pointer[i][j]->multiplyVectorized(matrix2->pointer[i][j]);
        }
    }
    return result;
}

OuterMatrix *OuterMatrix::multiplyNotVectorized(OuterMatrix *matrix2) {

    auto* result = new OuterMatrix(size);

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            result->pointer[i][j] = pointer[i][j]->multiplyNotVectorized(matrix2->pointer[i][j]);
        }
    }
    return result;
}

void OuterMatrix::initRandomFloat() {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            pointer[i][j]->initRandomFloat();
        }
    }
}
