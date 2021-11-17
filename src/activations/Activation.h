
//
// Created by Luecx on 10.11.2021.
//

#ifndef DIFFERENTIATION_SRC_ACITVATIONS_ACTIVATION_H_
#define DIFFERENTIATION_SRC_ACITVATIONS_ACTIVATION_H_

#include "../structures/Data.h"
struct Activation{
    virtual void apply      (Data *in        , Data *out                       ) = 0;
    virtual void backprop   (Data *out       , Data *in_grad   , Data *out_grad) = 0;
    virtual void logOverview() = 0;
};


#endif    // DIFFERENTIATION_SRC_ACITVATIONS_ACTIVATION_H_
