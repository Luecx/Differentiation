//
// Created by Luecx on 21.02.2021.
//
#include "Function.h"

#include <bitset>
#include <cmath>

void Linear::apply(Data* inp, Data* out) {
    assert(out->M == out->M);
    (*out) = (*inp);
}
void Linear::backprop(Data* out, Data* in_grad, Data* out_grad) {
    assert(out->M == out->M);
    (*in_grad) = (*out_grad);
}
void Linear::logOverview() {logging::write("Linear");}

void ReLU::apply(Data* inp, Data* out) {

    assert(inp->M == out->M);

    __m256    lower      = _mm256_set1_ps(0);

    float*    outputVals = out->values;
    float*    inputVals  = inp->values;

    const int size       = PARALLEL_SIZE_32_BIT(inp->M);

    for (int i = 0; i < size; i += 8) {
        __m256 in  = _mm256_load_ps(&(inputVals[i]));
        __m256 out = _mm256_max_ps(in, lower);

        _mm256_store_ps(&(outputVals[i]), out);
    }
    for (int i = size; i < inp->M; i++) {
        outputVals[i] = inputVals[i] < 0 ? 0 : inputVals[i];
    }
}
void ReLU::backprop(Data* out, Data* in_grad, Data* out_grad) {
    assert(out->M == out->M);

    static __m256    lower      = _mm256_set1_ps(0);
    static const int opera      = 30;    // _CMP_GT_OQ

    auto*          outputVals = (__m256*) out->values;
    auto*          inp_grad   = (__m256*) in_grad->values;
    auto*          oup_grad   = (__m256*) out_grad->values;

    const int        size       = PARALLEL_SIZE_32_BIT(out->M);

    for (int i = 0; i < size / 8; i++) {
        auto mask = _mm256_cmp_ps(outputVals[i], lower, opera);
        inp_grad[i] = _mm256_blendv_ps(lower, oup_grad[i], mask);
    }

    for (int i = size; i < out->M; i++) {
        (*in_grad)(i) = (*out)(i) > 0 ? (*out_grad)(i) : 0;
    }
}
void ReLU::logOverview() {logging::write("ReLU");}

void ClippedReLU::apply(Data* inp, Data* out) {

    assert(inp->M == out->M);

    __m256    lower      = _mm256_set1_ps(0);
    __m256    upper      = _mm256_set1_ps(max);

    float*    outputVals = out->values;
    float*    inputVals  = inp->values;

    const int size       = PARALLEL_SIZE_32_BIT(inp->M);

    for (int i = 0; i < size; i += 8) {
        __m256 in  = _mm256_load_ps(&(inputVals[i]));
        __m256 out = _mm256_min_ps(upper, _mm256_max_ps(in, lower));

        _mm256_store_ps(&(outputVals[i]), out);
    }
    for (int i = size; i < inp->M; i++) {
        outputVals[i] = inputVals[i] < 0 ? 0 : inputVals[i];
    }
}
void ClippedReLU::backprop(Data* out, Data* in_grad, Data* out_grad) {
    assert(out->M == out->M);

    static __m256    lower      = _mm256_set1_ps(0);
    static __m256    upper      = _mm256_set1_ps(max);
    static const int operaL      = 30;    // _CMP_GT_OQ
    static const int operaU      = 17;    // _CMP_LT_OQ

    auto*          outputVals = (__m256*) out->values;
    auto*          inp_grad   = (__m256*) in_grad->values;
    auto*          oup_grad   = (__m256*) out_grad->values;

    const int        size       = PARALLEL_SIZE_32_BIT(out->M);

    for (int i = 0; i < size / 8; i++) {
        auto maskLower = _mm256_cmp_ps(outputVals[i], lower, operaL); // mask all bits larger than 0
        auto maskUpper = _mm256_cmp_ps(outputVals[i], upper, operaU); // mask all bits lower than max
        inp_grad[i] = _mm256_blendv_ps(lower, oup_grad[i], maskLower);  // blend all which are larger than 0
        inp_grad[i] = _mm256_blendv_ps(lower, inp_grad[i], maskUpper);  // blend all which are lower than max
    }

    for (int i = size; i < out->M; i++) {
        (*in_grad)(i) = ((*out)(i) > 0 && (*out)(i) < max) ? (*out_grad)(i) : 0;
    }
}
void ClippedReLU::logOverview() {logging::write("ClippedReLU (" + std::to_string(max) + ")");}

void Sigmoid::apply(Data* inp, Data* out) {
    for(int i = 0; i < out->M; i++){
        (*out)(i) = 1.0 / (1 + expf(-(*inp)(i) * SIGMOID_SCALE));
    }
}
void Sigmoid::backprop(Data* out, Data* in_grad, Data* out_grad) {
    for(int i = 0; i < out->M; i++){
        (*in_grad)(i) = (*out_grad)(i) * ((*out)(i) * (1-(*out)(i))) * SIGMOID_SCALE;
    }
}
void Sigmoid::logOverview() {logging::write("Sigmoid (" + std::to_string(SIGMOID_SCALE) + ")");}




float MSE::apply(Data* out, Data* target) {
    assert(out->M == out->M);
    float loss = 0;
    for (int i = 0; i < out->M; i++) {
        loss += ((*out)(i) - (*target)(i)) * ((*out)(i) - (*target)(i));
    }
    return loss / out->M;
}
float MSE::backprop(Data* out, Data* target, Data* out_grad) {
    assert(out->M == out->M);
    float loss = 0;

    for (int i = 0; i < out->M; i++) {
        float diff     = ((*out)(i) - (*target)(i));
        (*out_grad)(i) = 2 * diff;
        loss += (diff * diff);
    }
    return loss / out->M;
}
void MSE::logOverview() {logging::write("MSE");}

