
//
// Created by Luecx on 10.11.2021.
//

#include "Sigmoid.h"

#include "../misc/logging.h"

#include <cmath>

void Sigmoid::apply(Data* inp, Data* out) {
    for (int i = 0; i < out->M; i++) {
        (*out)(i) = 1.0 / (1 + expf(-(*inp)(i) *SIGMOID_SCALE));
    }
}
void Sigmoid::backprop(Data* out, Data* in_grad, Data* out_grad) {
    for (int i = 0; i < out->M; i++) {
        (*in_grad)(i) = (*out_grad)(i) * ((*out)(i) * (1 - (*out)(i))) * SIGMOID_SCALE;
    }
}
void Sigmoid::logOverview() { logging::write("Sigmoid (" + std::to_string(SIGMOID_SCALE) + ")"); }

#include "Sigmoid.h"
