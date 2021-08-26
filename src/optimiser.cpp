//
// Created by Luecx on 23.02.2021.
//
#include "optimiser.h"

void Adam::init(std::vector<LayerInterface *> layers) {
    this->count = layers.size();
    this->layers = layers;

    first_moment_vector      = new  Data*[layers.size()*2];
    second_moment_vector     = new  Data*[layers.size()*2];
    for(int i = 0; i < layers.size(); i++){
        first_moment_vector[2*i]    = layers[i]->newWeightInstance();
        second_moment_vector[2*i]   = layers[i]->newWeightInstance();
        first_moment_vector[2*i+1]  = layers[i]->newBiasInstance();
        second_moment_vector[2*i+1] = layers[i]->newBiasInstance();
    }
}

void Adam::apply(Data *values, Data *gradient, Data *first_moment, Data *second_moment) {
#pragma omp parallel for schedule(auto) num_threads(UPDATE_THREADS)
    for(int i = 0; i < gradient->M * gradient->N; i++){
        (* first_moment)(i) = beta1 * (* first_moment)(i) + (1 - beta1) * (*gradient)(i);
        (*second_moment)(i) = beta2 * (*second_moment)(i) + (1 - beta2) * (*gradient)(i) * (*gradient)(i);

        double  first_moment_corrected = (* first_moment)(i) / (1 - pow(beta1, time));
        double second_moment_corrected = (*second_moment)(i) / (1 - pow(beta2, time));

        (*values)(i)  -= alpha * first_moment_corrected / (sqrt(second_moment_corrected) + eps);
        (*gradient)(i) = 0;
    }
}

Adam::~Adam() {
    for(int i = 0; i < count; i++){
        delete (first_moment_vector [2*i  ]);
        delete (second_moment_vector[2*i  ]);
        delete (first_moment_vector [2*i+1]);
        delete (second_moment_vector[2*i+1]);
    }

    delete[] (first_moment_vector);
    delete[] (second_moment_vector);

}

void Adam::apply(ThreadData *td, int batch_size) {
    float old_alpha = alpha;
    // correct alpha for the batch size
//    alpha *= sqrt(batch_size);
    for(int i = 0; i < count; i++){
        apply(layers.at(i)->getWeights(), td->weight_gradient[i], first_moment_vector[i*2+0], second_moment_vector[i*2+0]);
        apply(layers.at(i)->getBias()   , td->  bias_gradient[i], first_moment_vector[i*2+1], second_moment_vector[i*2+1]);
    }
    alpha = old_alpha;
}

void Adam::newEpoch() {
    time += 1;
}


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
    alpha /= sqrt(batch_size);
    for(int i = 0; i < count; i++){
        apply(layers.at(i)->getWeights(), td->weight_gradient[i]);
        apply(layers.at(i)->getBias()   , td->  bias_gradient[i]);
    }
    alpha = old_alpha;
}

void Gd::newEpoch() {
}
