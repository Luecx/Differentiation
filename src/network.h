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

struct Network{

private:
    std::vector<LayerInterface*> layers{};
    ThreadData *threadData[NN_THREADS]{};
    Loss        *loss      = nullptr;
    Optimiser   *optimiser = nullptr;

public:

    explicit Network(const std::vector<LayerInterface *> &layers);

//    Network(Network &&other);
//
//    Network(const Network &other) = delete;
//
//    Network& operator=(Network &&other);
//
//    Network& operator=(const Network &other) = delete;

    virtual ~Network();

    void setLoss(Loss* loss);

    void setOptimiser(Optimiser* optimiser);

    double batch(std::vector<Input> &inputs, std::vector<Data> &targets, int count = -1, bool train=true);

    double train(Input& input, Data& target);

    Data* evaluate(Input& input);

    Data* evaluate(Data* input);

    ThreadData* getThreadData(int thread);

    LayerInterface* getLayer(int layer);

    int layerCount();

    void loadWeights(const std::string &file);

    void saveWeights(const std::string &file);

    void newEpoch();

    void logOverview();

//    void train(std::vector<Position>& positions){
//
//    }


};

#endif //DIFFERENTIATION_NETWORK_H
