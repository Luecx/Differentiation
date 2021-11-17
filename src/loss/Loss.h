
//
// Created by Luecx on 10.11.2021.
//

#ifndef DIFFERENTIATION_SRC_LOSS_LOSS_H_
#define DIFFERENTIATION_SRC_LOSS_LOSS_H_
#include "../structures/Data.h"
struct Loss{
    virtual float apply      (Data *out       , Data *target                       ) = 0;
    virtual float backprop   (Data *out       , Data *target       , Data *out_grad) = 0;
    virtual void logOverview() = 0;
};

#endif    // DIFFERENTIATION_SRC_LOSS_LOSS_H_
