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

    Network(const std::vector<LayerInterface *> &layers);

    virtual ~Network();

    void setLoss(Loss* loss);

    void setOptimiser(Optimiser* optimiser);

    double batch(std::vector<Input> &inputs, std::vector<Data> &targets, int count = -1, bool train=true);

    Data* evaluate(Input& input);

    void loadWeights(const std::string &file){
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

    void saveWeights(const std::string &file){
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

    void newEpoch();

//    void train(std::vector<Position>& positions){
//
//    }


};

#endif //DIFFERENTIATION_NETWORK_H
