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

    __m256*          outputVals = (__m256*) out->values;
    __m256*          inp_grad   = (__m256*) in_grad->values;
    __m256*          oup_grad   = (__m256*) out_grad->values;

    const int        size       = PARALLEL_SIZE_32_BIT(out->M);

    for (int i = 0; i < size / 8; i++) {
        __m256 mask = _mm256_cmp_ps(outputVals[i], lower, opera);
        inp_grad[i] = _mm256_blendv_ps(lower, oup_grad[i], mask);
    }

    for (int i = size; i < out->M; i++) {
        (*in_grad)(i) = (*out)(i) > 0 ? (*out_grad)(i) : 0;
    }
}

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

    __m256*          outputVals = (__m256*) out->values;
    __m256*          inp_grad   = (__m256*) in_grad->values;
    __m256*          oup_grad   = (__m256*) out_grad->values;

    const int        size       = PARALLEL_SIZE_32_BIT(out->M);

    for (int i = 0; i < size / 8; i++) {
        __m256 maskLower = _mm256_cmp_ps(outputVals[i], lower, operaL); // mask all bits larger than 0
        __m256 maskUpper = _mm256_cmp_ps(outputVals[i], upper, operaU); // mask all bits lower than max
        inp_grad[i] = _mm256_blendv_ps(lower, oup_grad[i], maskLower);  // blend all which are larger than 0
        inp_grad[i] = _mm256_blendv_ps(lower, inp_grad[i], maskUpper);  // blend all which are lower than max
    }

    for (int i = size; i < out->M; i++) {
        (*in_grad)(i) = ((*out)(i) > 0 && (*out)(i) < max) ? (*out_grad)(i) : 0;
    }
}

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

float MSEmix::apply(Data* out, Data* target) {
    assert(out->M == target->M);
    assert(out->M == 1);
    float       loss      = 0;

    const float cpWeight  = 1 - wdlWeight;

    int16_t     tar       = target->get(0);
    float       output    = out->get(0);
    float       WDLtarget = 0.5;
    if (tar > 10000) {
        WDLtarget = 1;
        tar -= 20000;
    }
    if (tar < -10000) {
        WDLtarget = 0;
        tar += 20000;
    }

    float Ptarget = 1 / (1 + expf(-tar * SIGMOID_SCALE));

    loss          = powf(output - Ptarget, 2.0) * cpWeight + powf(output - WDLtarget, 2.0) * wdlWeight;

    return loss;
}

float MSEmix::backprop(Data* out, Data* target, Data* out_grad) {
    assert(out->M == target->M);
    assert(out->M == 1);
    float       loss      = 0;

    const float cpWeight  = 1 - wdlWeight;

    int16_t     tar       = target->get(0);
    float       output    = out->get(0);
    float       WDLtarget = 0.5;
    if (tar > 10000) {
        WDLtarget = 1;
        tar -= 20000;
    }
    if (tar < -10000) {
        WDLtarget = 0;
        tar += 20000;
    }

    float Ptarget = 1 / (1 + expf(-tar * SIGMOID_SCALE));

    loss          = powf(output - Ptarget, 2.0) * cpWeight + powf(output - WDLtarget, 2.0) * wdlWeight;

    float grad    = 0;
    grad += 2.0f * (output - Ptarget) * cpWeight;
    grad += 2.0f * (output - WDLtarget) * wdlWeight;

    out_grad->get(0) = grad;

    return loss;
}
