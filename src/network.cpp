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

//Network::Network(Network &&other){
//    this->loss = other.loss;
//    other.loss = nullptr;
////    this->threadData = other.threadData;
////    other.threadData = nullptr;
//    this->layers = other.layers;
//    this->optimiser = other.optimiser;
//    other.optimiser = nullptr;
//}

//Network::Network(const Network &other){
//    this->loss
//}

//Network& Network::operator=(Network &&other){
//    this->loss = other.loss;
//    other.loss = nullptr;
////    this->threadData = other.threadData;
////    other.threadData = nullptr;
//    this->layers = other.layers;
//    this->optimiser = other.optimiser;
//    other.optimiser = nullptr;
//    return *this;
//}

//Network& Network::operator=(const Network &other){
//
//}

void Network::setLoss(Loss *loss) {
    this->loss = loss;
}

void Network::setOptimiser(Optimiser *optimiser) {
    this->optimiser = optimiser;
    this->optimiser->init(layers);
}

Network::~Network() {
//    delete this->loss;
//    delete this->optimiser;
//    for (int i = 0; i < NN_THREADS; i++) {
//        delete threadData[i];
//        threadData[i] = nullptr;
//    }
//    this->loss = nullptr;
//    this->optimiser = nullptr;
}

double Network::batch(std::vector <Input> &inputs, std::vector <Data> &targets, int count, bool train) {
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


        if(train){
            // backward pass
            for(int l = layers.size()-1; l >= 1; l--){
                layers[l]->backprop(threadData[threadID]);
            }
            layers[0]->backprop(&inputs[i], threadData[threadID]);
        }

    }


    if(train) {
        merge_gradients(threadData);
        optimiser->apply(threadData[0], count);
    }

    return batchLoss / count;
}


double Network::train(Input& input, Data& target){

    layers[0]->apply(&input, threadData[0]);
    for (int l = 1; l < layers.size(); l++) {
        layers[l]->apply(threadData[0]);
    }

    // computing the loss
    double error = loss->backprop((threadData[0]->output[layers.size() - 1]), &target, (threadData[0]->output_gradient[layers.size() - 1]));

    // backward pass
    for (int l = layers.size() - 1; l >= 1; l--) {
        layers[l]->backprop(threadData[0]);
    }
    layers[0]->backprop(&input, threadData[0]);

    optimiser->apply(threadData[0], 1);
    return error;
}


Data* Network::evaluate(Input &input) {
    layers[0]->apply(&input, threadData[0]);
    for(int l = 1; l < layers.size(); l++){
        layers[l]->apply(threadData[0]);
    }
    return threadData[0]->output[layers.size()-1];
}

Data* Network::evaluate(Data *input) {
    layers[0]->apply(input, threadData[0]);
    for(int l = 1; l < layers.size(); l++){
        layers[l]->apply(threadData[0]);
    }
    return threadData[0]->output[layers.size()-1];
}

ThreadData *Network::getThreadData(int thread) {
    return threadData[thread];
}

LayerInterface* Network::getLayer(int layer){
    return layers[layer];
}

int Network::layerCount(){
    return layers.size();
}

void Network::newEpoch() {
    if(optimiser != nullptr){
        optimiser->newEpoch();
    }
}
