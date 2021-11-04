//
// Created by Luecx on 23.02.2021.
//

#ifndef DIFFERENTIATION_LAYER_H
#define DIFFERENTIATION_LAYER_H

#include "Data.h"
#include "Function.h"

class ThreadData;

struct LayerInterface{
protected:
    int layerID = 0;
public:
    virtual Data* getBias   () = 0;
    virtual Data* getWeights() = 0;

    virtual int getOutputSize() = 0;
    virtual int getInputSize()  = 0;

    virtual Activation* getActivationFunction() = 0;

    virtual Data* newOutputInstance() = 0;
    virtual Data* newWeightInstance() = 0;
    virtual Data* newBiasInstance() = 0;

    virtual void assignThreadData(ThreadData** td) = 0;
    virtual void apply(ThreadData* td) = 0;
    virtual void backprop(ThreadData* td) = 0;
    virtual void apply(Data *input, ThreadData* td) = 0;
    virtual void backprop(Data *input,ThreadData* td) = 0;
    virtual void apply(Input *input, ThreadData* td) = 0;
    virtual void backprop(Input *input,ThreadData* td) = 0;

    void assignID(int id){
        layerID = id;
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

    ThreadData(int ID, std::vector<LayerInterface*> layers);

    virtual ~ThreadData();
};


#endif //DIFFERENTIATION_LAYER_H
