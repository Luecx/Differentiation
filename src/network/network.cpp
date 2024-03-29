//
// Created by Luecx on 25.02.2021.
//

#include "network.h"

#include "../misc/logging.h"
#include "merge.h"

#include <omp.h>

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

Loss* Network::getLoss() const { return loss; }

Data* Network::getOutput(int threadID) const { return threadData[threadID]->output[layers.size() - 1]; }

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

    // reset used weights
    if(this->usedWeightColumns != nullptr){
        this->usedWeightColumns->clear();
    }

    // keeping track of the loss of this batch
    float batchLoss = 0;
    //making sure things run on multiple threads
#pragma omp parallel for schedule(auto) num_threads(NN_THREADS) reduction(+: batchLoss)
    for(int i = 0; i < count; i++){

        // create a new vector for which we will request the input
        const int threadID = omp_get_thread_num();

        // update the used weight columns if given
        if(this->usedWeightColumns != nullptr){
            int input_weight_cols = threadData[0]->weight_gradient[0]->getN();
            for(auto h:inputs[i].indices){
                this->usedWeightColumns->get(h % input_weight_cols) = 1;
            }
        }

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
        merge_gradients(threadData, this->usedWeightColumns);
        optimiser->apply(threadData[0], count);
    }

    return batchLoss / count;
}


double Network::train(Input& input, Data& target, bool update){

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
//
//    if (update)
//        optimiser->apply(threadData[0], 1);
    return error;
}


Data* Network::evaluate(Input &input) {
    layers[0]->apply(&input, threadData[0]);
    for(int l = 1; l < layers.size(); l++){
        layers[l]->apply(threadData[0]);
    }
    return getOutput();
}

Data* Network::evaluate(Data *input) {
    layers[0]->apply(input, threadData[0]);
    for(int l = 1; l < layers.size(); l++){
        layers[l]->apply(threadData[0]);
    }
    return threadData[0]->output[layers.size()-1];
}

void Network::trackSparseInputsOverBatch(bool value) {

    if(value == this->trackSparseInputs) return;

    if(!this->trackSparseInputs){
        // allocate the activated inputs
        this->usedWeightColumns = new Data(layers[0]->getWeights()->N);
    }else{
        delete this->usedWeightColumns;
        this->usedWeightColumns = nullptr;
    }
    this->trackSparseInputs = value;
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
void Network::loadWeights(const std::string& file) {
    FILE *f = fopen(file.c_str(), "rb");

    // figure out how many entries we will store
    uint64_t count = 0;
    for(LayerInterface* l:layers){
        count += l->getWeights()->size();
        count += l->getBias()->size();
    }

    uint64_t fileCount = 0;
    fread(&fileCount, sizeof(uint64_t), 1, f);
    assert(count == fileCount);

    for(LayerInterface* l:layers){
        fread(l->getWeights()->values, sizeof(float), l->getWeights()->size(), f);
        fread(l->getBias   ()->values, sizeof(float), l->getBias   ()->size(), f);
    }
    fclose(f);
}
void Network::saveWeights(const std::string& file) {
    FILE *f = fopen(file.c_str(), "wb");

    // figure out how many entries we will store
    uint64_t count = 0;
    for(LayerInterface* l:layers){
        count += l->getWeights()->size();
        count += l->getBias()->size();
    }

    fwrite(&count, sizeof(uint64_t), 1, f);
    for(LayerInterface* l:layers){
        fwrite(l->getWeights()->values, sizeof(float), l->getWeights()->size(), f);
        fwrite(l->getBias   ()->values, sizeof(float), l->getBias   ()->size(), f);
    }
    fclose(f);
}
void Network::logOverview() {
    logging::write("----------------- Network overview -----------------------");
    logging::write("arch:");
    for(LayerInterface* l:layers){
        logging::write("    " + std::to_string(l->getInputSize()) + " --> " + std::to_string(l->getOutputSize()));
    }
    logging::write("optimiser: ","");
    this->optimiser->logOverview();

    logging::write("loss: ","");
    this->loss->logOverview();

}
