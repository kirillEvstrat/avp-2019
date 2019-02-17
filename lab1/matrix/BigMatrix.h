//
// Created by Rostislav Pekhovsky on 2019-02-07.
//

#ifndef LAB1_1_OUTER_MATRIX_H
#define LAB1_1_OUTER_MATRIX_H

#include "InnerMatrix.h"

class BigMatrix {
    InnerMatrix*** pointer;
    int size;
    int innerMatrixSize;

public:
    explicit BigMatrix(int size, int innerMatrixSize);
    BigMatrix* MultiplyVectorized(const BigMatrix *matrix2);
    BigMatrix* MultiplyNotVectorized(const BigMatrix *matrix2);
    BigMatrix* MultiplyManuallyVectorized(const BigMatrix* matrix);

    void ShowStdout();
    void InitRandomFloat();

    bool operator==(const BigMatrix &rhs) const;
    bool operator!=(const BigMatrix &rhs) const;
};

#endif //LAB1_1_OUTERMATRIX_H

