
//
// Created by Luecx on 10.11.2021.
//

#ifndef DIFFERENTIATION_SRC_LOSS_MSE_H_
#define DIFFERENTIATION_SRC_LOSS_MSE_H_

#include "../structures/Data.h"
#include "Loss.h"
struct MSE : Loss{
    float apply     (Data *out, Data *target) override;
    float backprop  (Data *out, Data *target, Data *out_grad) override;
    void logOverview() override;
};

#endif    // DIFFERENTIATION_SRC_LOSS_MSE_H_
