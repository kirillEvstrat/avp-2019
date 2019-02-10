#include <iostream>
#include <random>
#include <ctime>
#include <random>
#include <iomanip>
#include "matrix/BigMatrix.h"

using namespace std::chrono;

unsigned long t1, t2, t3, t4;

const int BIG_MATRIX_SIZE = 200;
const int INNER_MATRIX_SIZE = 8;

int main() {

    auto* bigMatrix1 = new BigMatrix(BIG_MATRIX_SIZE, INNER_MATRIX_SIZE);
    auto* bigMatrix2 = new BigMatrix(BIG_MATRIX_SIZE, INNER_MATRIX_SIZE);
    bigMatrix1->InitRandomFloat();
    bigMatrix2->InitRandomFloat();

    t3 = clock();
    auto* bigMatrixRes1 = bigMatrix1->MultiplyVectorized(bigMatrix2);
    t4 = clock();

    t1 = clock();
    auto* bigMatrixRes2 = bigMatrix1->MultiplyNotVectorized(bigMatrix2);
    t2 = clock();


    std::cout <<  std::setprecision (20) << "Time not vectorized: " << float(t2 - t1) / CLOCKS_PER_SEC << std::endl;
    std::cout <<  std::setprecision (20) << "Time     vectorized: " << float(t4 - t3) / CLOCKS_PER_SEC  << std::endl;

    return 0;
}