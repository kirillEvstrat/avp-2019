//
// Created by Rostislav Pekhovsky on 2019-02-07.
//

#include "random.h"

float RandomFloat(float startVal, float endVal) {
    std::mt19937 gen1(static_cast<unsigned int>(clock()));
    std::uniform_real_distribution<> urd(startVal, endVal);
    return static_cast<float>(urd(gen1));
}

float RandomFloat() {
    return RandomFloat(-1000, 1000);
}