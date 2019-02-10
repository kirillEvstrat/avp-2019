//
// Created by Rostislav Pekhovsky on 2019-02-07.
//

#ifndef LAB1_1_INNER_MATRIX_H
#define LAB1_1_INNER_MATRIX_H

#include "../random/random.h"

class InnerMatrix {
    float **pointer;
    int size;

public:

    explicit InnerMatrix(int size);
    InnerMatrix* multiplyVectorized(InnerMatrix* matrix2);
    InnerMatrix* multiplyNotVectorized(InnerMatrix* matrix2);
    void initRandomFloat();
};



#endif //LAB1_1_INNER_MATRIX_H
