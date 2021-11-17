
//
// Created by Luecx on 10.11.2021.
//

#include "Linear.h"

#include "../misc/logging.h"

void Linear::apply(Data* inp, Data* out) {
    assert(out->M == out->M);
    (*out) = (*inp);
}
void Linear::backprop(Data* out, Data* in_grad, Data* out_grad) {
    assert(out->M == out->M);
    (*in_grad) = (*out_grad);
}
void Linear::logOverview() { logging::write("Linear"); }
