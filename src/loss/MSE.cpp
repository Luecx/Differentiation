
//
// Created by Luecx on 10.11.2021.
//

#include "MSE.h"

#include "../misc/logging.h"

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
