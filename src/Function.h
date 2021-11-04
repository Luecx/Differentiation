//
// Created by Luecx on 21.02.2021.
//

#ifndef DIFFERENTIATION_FUNCTION_H
#define DIFFERENTIATION_FUNCTION_H
#include "Data.h"
#include "logging.h"
#include "util.h"

#include <cmath>

struct Activation{
    virtual void apply      (Data *in        , Data *out                       ) = 0;
    virtual void backprop   (Data *out       , Data *in_grad   , Data *out_grad) = 0;
    virtual void logOverview() = 0;
};

struct ReLU : Activation{
    void apply      (Data *inp       , Data *out) override;
    void backprop   (Data *out       , Data *in_grad   , Data *out_grad) override;
    void logOverview() override;
};

struct ClippedReLU : Activation{
    float max = 127;
    void apply      (Data *inp       , Data *out) override;
    void backprop   (Data *out       , Data *in_grad   , Data *out_grad) override;
    void logOverview() override;
};

struct Linear : Activation {
    void apply      (Data *inp       , Data *out) override;
    void backprop   (Data *out       , Data *in_grad   , Data *out_grad) override;
    void logOverview() override;
};

struct Sigmoid : Activation{
    void apply      (Data *inp       , Data *out) override;
    void backprop   (Data *out       , Data *in_grad   , Data *out_grad) override;
    void logOverview() override;
};

struct Loss{
    virtual float apply      (Data *out       , Data *target                       ) = 0;
    virtual float backprop   (Data *out       , Data *target       , Data *out_grad) = 0;
    virtual void logOverview() = 0;
};

struct MSE : Loss{
    float apply     (Data *out, Data *target) override;
    float backprop  (Data *out, Data *target, Data *out_grad) override;
    void logOverview() override;
};


#endif //DIFFERENTIATION_FUNCTION_H
