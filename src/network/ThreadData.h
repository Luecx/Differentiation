
//
// Created by Luecx on 10.11.2021.
//

#ifndef DIFFERENTIATION_SRC_THREADDATA_H_
#define DIFFERENTIATION_SRC_THREADDATA_H_

#include "../structures/Data.h"

struct LayerInterface;

class ThreadData {

    public:

    Data** output;
    Data** output_gradient;
    Data** weight_gradient;
    Data** bias_gradient;

    const int threadID;
    const int count;

    ThreadData(int ID, std::vector<LayerInterface*> layers);

    virtual ~ThreadData();
};

#endif    // DIFFERENTIATION_SRC_THREADDATA_H_
