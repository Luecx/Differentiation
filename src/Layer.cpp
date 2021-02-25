//
// Created by Luecx on 25.02.2021.
//
#include "Layer.h"

ThreadData::ThreadData(int ID, std::vector<LayerInterface *> layers) : count(layers.size()), threadID(ID){
    output              = new  Data*[layers.size()];
    output_gradient     = new  Data*[layers.size()];
    weight_gradient     = new  Data*[layers.size()];
    bias_gradient       = new  Data*[layers.size()];
    for(int i = 0; i < layers.size(); i++){
        output[i]           = layers[i]->newOutputInstance();
        output_gradient[i]  = layers[i]->newOutputInstance();
        weight_gradient[i]  = layers[i]->newWeightInstance();
        bias_gradient[i]    = layers[i]->newBiasInstance();
    }
}

ThreadData::~ThreadData() {
    for(int i = 0; i < count; i++){
        delete (output[i]);
        delete (output_gradient[i]);
        delete (weight_gradient[i]);
        delete (bias_gradient[i]);
    }

    delete[] (output);
    delete[] (output_gradient);
    delete[] (weight_gradient);
    delete[] (bias_gradient);

}
