//
// Created by Luecx on 21.02.2021.
//

#ifndef DIFFERENTIATION_FUNCTION_H
#define DIFFERENTIATION_FUNCTION_H
#include "Data.h"
#include "util.h"
#include "DenseLayer.h"


template<Dimension M>
struct Activation{
    virtual void apply      (Data<M> *in        , Data<M> *out                          ) = 0;
    virtual void backprop   (Data<M> *out       , Data<M> *in_grad   , Data<M> *out_grad) = 0;
};

template<Dimension M>
struct ReLU : Activation<M>{
    void apply      (Data<M> *inp       , Data<M> *out) override {
        __m256 lower = _mm256_set1_ps(0);

        float* outputVals = out->values;
        float*  inputVals = inp->values;

        constexpr int size = PARALLEL_SIZE_32_BIT(M);

        for(int i = 0; i < size; i+=8){
            __m256 in  = _mm256_load_ps(&(inputVals[i]));
            __m256 out = _mm256_max_ps(in, lower);

            _mm256_store_ps(&(outputVals[i]), out);
        }
        for (int i = size; i < M; i++){
            outputVals[i] = inputVals[i] < 0 ? 0:inputVals[i];
        }

    }
    void backprop   (Data<M> *out       , Data<M> *in_grad   , Data<M> *out_grad) override {
        static          __m256 lower = _mm256_set1_ps(0);
        static const    int    opera  = 30;                  // _CMP_GT_OQ

        __m256* outputVals  = (__m256*) out     ->values;
        __m256* inp_grad    = (__m256*) in_grad ->values;
        __m256* oup_grad    = (__m256*) out_grad->values;

        constexpr int size = PARALLEL_SIZE_32_BIT(M);

        for (int i = 0; i < size / 8; i++){
            __m256 mask = _mm256_cmp_ps (outputVals[i], lower, opera);
            inp_grad[i] = _mm256_blendv_ps( lower,oup_grad[i], mask);
        }

        for (int i = size; i < M; i++){
            (*in_grad)(i) = (*out)(i) > 0 ? (*out_grad)(i) : 0;
        }
    }
};


template <Dimension M>
struct Loss{
    virtual float apply      (Data<M> *out       , Data<M> *target                          ) = 0;
    virtual float backprop   (Data<M> *out       , Data<M> *target       , Data<M> *out_grad) = 0;
};

template <Dimension M>
struct MSE : Loss<M>{
    float apply     (Data<M> *out, Data<M> *target) override {
        float loss = 0;
        for(int i = 0; i < M; i++){
            loss += ((*out)(i) - (*target)(i)) * ((*out)(i) - (*target)(i));
        }
        return loss / M;
    }
    float backprop  (Data<M> *out, Data<M> *target, Data<M> *out_grad) override {
        float loss = 0;
        for(int i = 0; i < M; i++){
            float diff = ((*out)(i) - (*target)(i));
            (*out_grad)(i) = 2 * diff;
            loss += (diff * diff);
        }
        return loss / M;
    }
};


#endif //DIFFERENTIATION_FUNCTION_H
