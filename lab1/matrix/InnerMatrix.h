//
// Created by Rostislav Pekhovsky on 2019-02-07.
//

#ifndef LAB1_1_INNER_MATRIX_H
#define LAB1_1_INNER_MATRIX_H

#include <iostream>
#include <xmmintrin.h>

#include "../random/random.h"

class InnerMatrix {
    float **pointer;
    int size;

public:

    explicit InnerMatrix(int size);

    InnerMatrix* MultiplyVectorized(const InnerMatrix* matrix);
    InnerMatrix* MultiplyNotVectorized(const InnerMatrix* matrix);
    InnerMatrix* MultiplyManuallyVectorized(const InnerMatrix* matrix);

    void AddVectorized(const InnerMatrix* matrix);
    void AddNotVectorized(const InnerMatrix* matrix);
    float **GetPointer() const;

    void ShowStdout();
    void InitRandomFloat();

    bool operator==(const InnerMatrix &rhs) const;

    bool operator!=(const InnerMatrix &rhs) const;
};



#endif //LAB1_1_INNER_MATRIX_H
