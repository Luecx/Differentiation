//
// Created by Luecx on 21.02.2021.
//
#include <cmath>
#include "Function.h"


void ReLU::apply(Data *inp, Data *out) {

    assert(inp->M == out->M);

    __m256 lower = _mm256_set1_ps(0);

    float* outputVals = out->values;
    float*  inputVals = inp->values;

    const int size = PARALLEL_SIZE_32_BIT(inp->M);

    for(int i = 0; i < size; i+=8){
        __m256 in  = _mm256_load_ps(&(inputVals[i]));
        __m256 out = _mm256_max_ps(in, lower);

        _mm256_store_ps(&(outputVals[i]), out);
    }
    for (int i = size; i < inp->M; i++){
        outputVals[i] = inputVals[i] < 0 ? 0:inputVals[i];
    }

}

void ReLU::backprop(Data *out, Data *in_grad, Data *out_grad) {
    assert(out->M == out->M);

    static          __m256 lower = _mm256_set1_ps(0);
    static const    int    opera  = 30;                  // _CMP_GT_OQ

    __m256* outputVals  = (__m256*) out     ->values;
    __m256* inp_grad    = (__m256*) in_grad ->values;
    __m256* oup_grad    = (__m256*) out_grad->values;

    const int size = PARALLEL_SIZE_32_BIT(out->M);

    for (int i = 0; i < size / 8; i++){
        __m256 mask = _mm256_cmp_ps (outputVals[i], lower, opera);
        inp_grad[i] = _mm256_blendv_ps( lower,oup_grad[i], mask);
    }

    for (int i = size; i < out->M; i++){
        (*in_grad)(i) = (*out)(i) > 0 ? (*out_grad)(i) : 0;
    }
}

void Linear::apply(Data *inp, Data *out) {
    assert(out->M == out->M);
    (*out) = (*inp);
}

void Linear::backprop(Data *out, Data *in_grad, Data *out_grad) {
    assert(out->M == out->M);
    (*in_grad) = (*out_grad);
}

void Sigmoid::apply(Data *inp, Data *out) {
    assert(out->M == out->M);
    for(int i = 0; i < out->M; i++){
        (*out)(i) = 1.0 / (1 + exp(-(*inp)(i)));
    }
}

void Sigmoid::backprop(Data *out, Data *in_grad, Data *out_grad) {
    assert(out->M == out->M);
    for(int i = 0; i < out->M; i++){
        (*in_grad)(i) = (*out_grad)(i) * ((*out)(i) * (1-(*out)(i)));
    }
}

float MSE::apply(Data *out, Data *target) {
    assert(out->M == out->M);
    float loss = 0;
    for(int i = 0; i < out->M; i++){
        loss += ((*out)(i) - (*target)(i)) * ((*out)(i) - (*target)(i));
    }
    return loss / out->M;
}

float MSE::backprop(Data *out, Data *target, Data *out_grad) {
    assert(out->M == out->M);
    float loss = 0;

    for(int i = 0; i < out->M; i++){
        float diff = ((*out)(i) - (*target)(i));
        (*out_grad)(i) = 2 * diff;
        loss += (diff * diff);
    }
    return loss / out->M;
}
