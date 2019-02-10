//
// Created by Rostislav Pekhovsky on 2019-02-07.
//

#ifndef LAB1_1_OUTER_MATRIX_H
#define LAB1_1_OUTER_MATRIX_H

#include "InnerMatrix.h"

class OuterMatrix {
    InnerMatrix*** pointer;
    int size;

public:
    explicit OuterMatrix(int size);
    OuterMatrix* multiplyVectorized(OuterMatrix* matrix2);
    OuterMatrix* multiplyNotVectorized(OuterMatrix* matrix2);
    void initRandomFloat();
};

#endif //LAB1_1_OUTERMATRIX_H

