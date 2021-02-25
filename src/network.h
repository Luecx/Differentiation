//
// Created by Luecx on 24.02.2021.
//

#ifndef DIFFERENTIATION_NETWORK_H
#define DIFFERENTIATION_NETWORK_H

#include "Layer.h"
#include "config.h"
#include "Data.h"
#include "Function.h"
#include "optimiser.h"

//template<typename D, typename Generator>
struct Network{

private:
    std::vector<LayerInterface*> layers{};
    ThreadData *threadData[NN_THREADS]{};
    Loss        *loss      = nullptr;
    Optimiser   *optimiser = nullptr;

public:

    Network(const std::vector<LayerInterface *> &layers);

    void setLoss(Loss* loss);

    void setOptimiser(Optimiser* optimiser);

    virtual ~Network();

    double batch(std::vector<Input> &inputs, std::vector<Data> &targets);

    void newEpoch();

//    void train(std::vector<Position>& positions){
//
//    }


};

#endif //DIFFERENTIATION_NETWORK_H
