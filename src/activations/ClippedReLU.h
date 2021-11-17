
//
// Created by Luecx on 10.11.2021.
//

#ifndef DIFFERENTIATION_SRC_ACITVATIONS_CLIPPEDRELU_H_
#define DIFFERENTIATION_SRC_ACITVATIONS_CLIPPEDRELU_H_

#include "../structures/Data.h"
#include "Activation.h"

struct ClippedReLU : Activation{
    float max = 127;
    void apply      (Data *inp       , Data *out) override;
    void backprop   (Data *out       , Data *in_grad   , Data *out_grad) override;
    void logOverview() override;
};


#endif    // DIFFERENTIATION_SRC_ACITVATIONS_CLIPPEDRELU_H_
