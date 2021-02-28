//
// Created by Luecx on 25.02.2021.
//

#include <omp.h>
#include "network.h"
#include "merge.h"

Network::Network(const std::vector<LayerInterface *> &layers) : layers(layers) {
    for (int i = 0; i < NN_THREADS; i++) {
        threadData[i] = new ThreadData{i, layers};
    }
    int index = 0;
    for (LayerInterface *l:layers) {
        l->assignID(index);
        l->assignThreadData(threadData);
        index ++;
    }
}

void Network::setLoss(Loss *loss) {
    this->loss = loss;
}

void Network::setOptimiser(Optimiser *optimiser) {
    this->optimiser = optimiser;
    this->optimiser->init(layers);
}

Network::~Network() {
    delete this->loss;
    delete this->optimiser;
    for (int i = 0; i < NN_THREADS; i++) {
        delete threadData[i];
    }
}

double Network::batch(std::vector <Input> &inputs, std::vector <Data> &targets, int count) {
    assert(inputs.size() == targets.size());
    assert(loss != nullptr);
    assert(optimiser != nullptr);

    if(count < 0) count = inputs.size();
    if(count > inputs.size()) count = inputs.size();


    // keeping track of the loss of this batch
    float batchLoss = 0;
    //making sure things run on multiple threads
#pragma omp parallel for schedule(auto) num_threads(NN_THREADS) reduction(+: batchLoss)
    for(int i = 0; i < count; i++){
        // create a new vector for which we will request the input

        const int threadID = omp_get_thread_num();

        // forward pass
        layers[0]->apply(&inputs[i], threadData[threadID]);
        for(int l = 1; l < layers.size(); l++){
            layers[l]->apply(threadData[threadID]);
        }

        // computing the loss
        batchLoss += loss->backprop((threadData[threadID]->output         [layers.size() - 1]), &targets[i],
                                    (threadData[threadID]->output_gradient[layers.size() - 1]));



        // backward pass
        for(int l = layers.size()-1; l >= 1; l--){
            layers[l]->backprop(threadData[threadID]);
        }
        layers[0]->backprop(&inputs[i], threadData[threadID]);
    }

    merge_gradients(threadData);
    optimiser->apply(threadData[0], count);

    return batchLoss / count;
}


Data* Network::evaluate(Input &input) {
    layers[0]->apply(&input, threadData[0]);
    for(int l = 1; l < layers.size(); l++){
        layers[l]->apply(threadData[0]);
    }
    return threadData[0]->output[layers.size()-1];
}


void Network::newEpoch() {
    if(optimiser != nullptr){
        optimiser->newEpoch();
    }
}
