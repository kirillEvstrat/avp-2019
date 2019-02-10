#include <iostream>
#include <random>
#include <ctime>
#include <random>
#include <iomanip>
#include "matrix/OuterMatrix.h"

using namespace std::chrono;

const int ARRAY_SIZE = 100000000;

unsigned long t1, t2, t3, t4;

//
//float *multiplyNotVectorized(const float * a, const float * b, int size) {
//
//    auto * c = new float[size];
//
//    t1 = clock();
//    #pragma novector
//    for (int i = 0; i < size; i++) {
//        c[i] += b[i] * a[i];
//    }
//    t2 = clock();
//    return c;
//}
//
//
//float *multiplyVectorized(const float* __restrict a, const float* __restrict b, int size) {
//
//    auto * __restrict c = new float[size];
//
//    t3 = clock();
//    #pragma vector always
//    for (int i = 0; i < size; i++) {
//        c[i] += b[i] * a[i];
//    }
//    t4 = clock();
//
//    return c;
//}
//
//void init(float *arr, int size) {
//    for (int i = 0; i < size; i+=100) {
//        float a = randomFloat();
//        for (int j = 0; j < 100; j++) {
//            arr[i + j] = a;
//        }
//    }
//}

int main() {

    auto* outerMatrix1 = new OuterMatrix(132);
    auto* outerMatrix2 = new OuterMatrix(132);
    outerMatrix1->initRandomFloat();
    outerMatrix2->initRandomFloat();

    t3 = clock();
    auto* res1 = outerMatrix1->multiplyVectorized(outerMatrix2);
    t4 = clock();

    outerMatrix1->initRandomFloat();
    outerMatrix2->initRandomFloat();

    t1 = clock();
    auto* res2 = outerMatrix1->multiplyNotVectorized(outerMatrix2);
    t2 = clock();



    std::cout << "Time not vectorized: " << double(t2 - t1) / CLOCKS_PER_SEC << std::endl;
    std::cout << "Time     vectorized: " << double(t4 - t3) / CLOCKS_PER_SEC  << std::endl;



    return 0;
}