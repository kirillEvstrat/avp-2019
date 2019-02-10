//
// Created by Rostislav Pekhovsky on 2019-02-07.
//

#ifndef LAB1_1_INNER_MATRIX_H
#define LAB1_1_INNER_MATRIX_H

#include <iostream>
#include "../random/random.h"

class InnerMatrix {
    float **pointer;
    int size;

    InnerMatrix* Multiply(InnerMatrix* matrix);
public:

    explicit InnerMatrix(int size);
    InnerMatrix* MultiplyVectorized(InnerMatrix* matrix);
    InnerMatrix* MultiplyNotVectorized(InnerMatrix* matrix);
    void AddVectorized(InnerMatrix* matrix);
    void AddNotVectorized(InnerMatrix* matrix);
    float **GetPointer() const;

    void ShowStdout();
    void InitRandomFloat();
};



#endif //LAB1_1_INNER_MATRIX_H
