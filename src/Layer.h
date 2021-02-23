//
// Created by Luecx on 23.02.2021.
//

#ifndef DIFFERENTIATION_LAYER_H
#define DIFFERENTIATION_LAYER_H

#include "Data.h"

static int layer_id_counter;

class ThreadData;

struct LayerInterface{
protected:
    int layerID = 0;
public:
    virtual Data* getBias   () = 0;
    virtual Data* getWeights() = 0;

    virtual int getOutputSize() = 0;
    virtual int getInputSize()  = 0;

    virtual Data* newOutputInstance() = 0;
    virtual Data* newWeightInstance() = 0;
    virtual Data* newBiasInstance() = 0;

    virtual void assignThreadData(ThreadData** td) = 0;
    void assignID(){
        layerID = layer_id_counter ++;
    }
};


class ThreadData {
private:

public:

    Data** output;
    Data** output_gradient;
    Data** weight_gradient;
    Data**   bias_gradient;

    const int threadID;
    const int count;

    ThreadData(int ID, std::vector<LayerInterface*> layers) : count(layers.size()), threadID(ID){
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

    virtual ~ThreadData() {
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
};


#endif //DIFFERENTIATION_LAYER_H
