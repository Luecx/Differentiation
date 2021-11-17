
//
// Created by Luecx on 17.11.2021.
//
#include "../misc/logging.h"
#include "../network/ThreadData.h"

#include "GD.h"

void Gd::init(std::vector<LayerInterface *> layers) {
    this->count = layers.size();
    this->layers = layers;
}

void Gd::apply(Data *values, Data *gradient) {
#pragma omp parallel for schedule(auto) num_threads(UPDATE_THREADS)
    for(int i = 0; i < gradient->M * gradient->N; i++){
        (*values)(i)  -= alpha * (*gradient)(i);
        (*gradient)(i) = 0;
    }
}

Gd::~Gd() = default;

void Gd::apply(ThreadData *td, int batch_size) {
    float old_alpha = alpha;
    // correct alpha for the batch size
//    alpha *= sqrt(batch_size);
//    alpha /= sqrt(batch_size);
    for(int i = 0; i < count; i++){
        apply(layers.at(i)->getWeights(), td->weight_gradient[i]);
        apply(layers.at(i)->getBias()   , td->  bias_gradient[i]);
    }
//    alpha = old_alpha;
}
void Gd::logOverview() {
    logging::write("GD:");
    logging::write("    alpha: "+std::to_string(alpha));
}
void Gd::newEpoch() {
}