//
// Created by Luecx on 21.02.2021.
//

#ifndef DIFFERENTIATION_FUNCTION_H
#define DIFFERENTIATION_FUNCTION_H
#include "Data.h"
#include "util.h"

#include <cmath>

struct Activation{
    virtual void apply      (Data *in        , Data *out                       ) = 0;
    virtual void backprop   (Data *out       , Data *in_grad   , Data *out_grad) = 0;
};

struct ReLU : Activation{
    void apply      (Data *inp       , Data *out) override;
    void backprop   (Data *out       , Data *in_grad   , Data *out_grad) override;
};

struct ClippedReLU : Activation{
    float max = 127;
    void apply      (Data *inp       , Data *out) override;
    void backprop   (Data *out       , Data *in_grad   , Data *out_grad) override;
};

struct Linear : Activation {
    void apply      (Data *inp       , Data *out) override;
    void backprop   (Data *out       , Data *in_grad   , Data *out_grad) override;
};

struct Sigmoid : Activation{
    void apply      (Data *inp       , Data *out) override {
        assert(out->M == out->M);
        for(int i = 0; i < out->M; i++){
            (*out)(i) = 1.0 / (1 + expf(-(*inp)(i) * SIGMOID_SCALE));
        }
    }
    void backprop   (Data *out       , Data *in_grad   , Data *out_grad) override{
        assert(out->M == out->M);
        for(int i = 0; i < out->M; i++){
            (*in_grad)(i) = (*out_grad)(i) * ((*out)(i) * (1-(*out)(i))) * SIGMOID_SCALE;
        }
    }
};

struct Loss{
    virtual float apply      (Data *out       , Data *target                       ) = 0;
    virtual float backprop   (Data *out       , Data *target       , Data *out_grad) = 0;
};

struct MSE : Loss{
    float apply     (Data *out, Data *target) override;
    float backprop  (Data *out, Data *target, Data *out_grad) override;
};

struct MSEmix : Loss{

    float wdlWeight = 0.5;

    float apply     (Data* out, Data *target) override;
    float backprop  (Data *out, Data *target, Data *out_grad) override;
};

#endif //DIFFERENTIATION_FUNCTION_H
