//
// Created by Luecx on 21.02.2021.
//

#ifndef DIFFERENTIATION_FUNCTION_H
#define DIFFERENTIATION_FUNCTION_H
#include "Data.h"
#include "util.h"


struct Activation{
    virtual void apply      (Data *in        , Data *out                       ) = 0;
    virtual void backprop   (Data *out       , Data *in_grad   , Data *out_grad) = 0;
};

struct ReLU : Activation{
    void apply      (Data *inp       , Data *out);
    void backprop   (Data *out       , Data *in_grad   , Data *out_grad);
};

struct Linear : Activation {
    void apply      (Data *inp       , Data *out);
    void backprop   (Data *out       , Data *in_grad   , Data *out_grad);
};

struct Sigmoid : Activation{
    void apply      (Data *inp       , Data *out);
    void backprop   (Data *out       , Data *in_grad   , Data *out_grad);
};

struct Loss{
    virtual float apply      (Data *out       , Data *target                       ) = 0;
    virtual float backprop   (Data *out       , Data *target       , Data *out_grad) = 0;
};

struct MSE : Loss{
    float apply     (Data *out, Data *target) override;
    float backprop  (Data *out, Data *target, Data *out_grad) override;
};


#endif //DIFFERENTIATION_FUNCTION_H
