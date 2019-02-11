#include <iostream>
#include <random>
#include <ctime>
#include <random>
#include <iomanip>
#include "matrix/BigMatrix.h"

using namespace std::chrono;



const int BIG_MATRIX_SIZE = 1;
const int INNER_MATRIX_SIZE = 8;

int main() {
    unsigned long t1, t2, t3, t4, t5, t6;

    auto* bigMatrix1 = new BigMatrix(BIG_MATRIX_SIZE, INNER_MATRIX_SIZE);
    auto* bigMatrix2 = new BigMatrix(BIG_MATRIX_SIZE, INNER_MATRIX_SIZE);

    bigMatrix1->InitRandomFloat();
    bigMatrix2->InitRandomFloat();

    t3 = clock();
    auto* bigMatrixRes1 = bigMatrix1->MultiplyVectorized(bigMatrix2);
    t4 = clock();

    bigMatrix1->InitRandomFloat();
    bigMatrix2->InitRandomFloat();

    t1 = clock();
    auto* bigMatrixRes2 = bigMatrix1->MultiplyNotVectorized(bigMatrix2);
    t2 = clock();

    bigMatrix1->InitRandomFloat();
    bigMatrix2->InitRandomFloat();

    t5 = clock();
    auto* bigMatrixRes3 = bigMatrix1->MultiplyManuallyVectorized(bigMatrix2);
    t6 = clock();

    std::cout << "Big matrix 1: " << std::endl;
    bigMatrix1->ShowStdout();

    std::cout << "Big matrix 2: " << std::endl;
    bigMatrix2->ShowStdout();

    std::cout << "Res matrix 1: " << std::endl;
    bigMatrixRes1->ShowStdout();

    std::cout << "Res matrix 2: " << std::endl;
    bigMatrixRes2->ShowStdout();

    std::cout << "Res matrix 3: " << std::endl;
    bigMatrixRes3->ShowStdout();

    std::cout <<  std::setprecision (30) << "Time      not vectorized: " << float(t2 - t1) / CLOCKS_PER_SEC << std::endl;
    std::cout <<  std::setprecision (30) << "Time          vectorized: " << float(t4 - t3) / CLOCKS_PER_SEC  << std::endl;
    std::cout <<  std::setprecision (30) << "Time manually vectorized: " << float(t6 - t5) / CLOCKS_PER_SEC  << std::endl;


    delete bigMatrix1;
    delete bigMatrix2;
    delete bigMatrixRes1;
    return 0;
}