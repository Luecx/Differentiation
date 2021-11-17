//
// Created by Luecx on 24.02.2021.
//

#ifndef DIFFERENTIATION_NETWORK_H
#define DIFFERENTIATION_NETWORK_H

#include "../activations/Activation.h"
#include "../loss/Loss.h"
#include "../layers/Layer.h"
#include "../misc/config.h"
#include "../optimiser/optimiser.h"
#include "../structures/Data.h"
#include "ThreadData.h"

struct Network{

private:
    std::vector<LayerInterface*> layers{};
    ThreadData *threadData[NN_THREADS]{};
    Loss        *loss      = nullptr;
    Optimiser   *optimiser = nullptr;

    // true if the inputs will be tracked and used for efficient updates
    bool trackSparseInputs = false;
    Data*                        usedWeightColumns = nullptr;

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

    void            setLoss(Loss* loss);

    void            setOptimiser(Optimiser* optimiser);

    Loss*           getLoss() const;

    Data*           getOutput(int threadID=0) const;

    void            trackSparseInputsOverBatch(bool value);

    double          batch(std::vector<Input>& inputs, std::vector<Data>& targets, int count = -1, bool train = true);

    double          train(Input& input, Data& target, bool update=true);

    Data*           evaluate(Input& input);

    Data*           evaluate(Data* input);

    ThreadData*     getThreadData(int thread);

    LayerInterface* getLayer(int layer);

    int             layerCount();

    void            loadWeights(const std::string& file);

    void            saveWeights(const std::string& file);

    void            newEpoch();

    void            logOverview();
};

#endif //DIFFERENTIATION_NETWORK_H
