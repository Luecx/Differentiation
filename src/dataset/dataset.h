
//
// Created by Luecx on 27.11.2021.
//

#ifndef BINARYPOSITIONWRAPPER_SRC_DATASET_DATASET_H_
#define BINARYPOSITIONWRAPPER_SRC_DATASET_DATASET_H_

#include "../position/position.h"
#include "header.h"

#include <vector>

struct DataSet {

    Header                header {};
    std::vector<Position> positions {};

    void addData(DataSet& other){
        positions.insert(std::end(positions), std::begin(other.positions), std::end(other.positions));
    }
};

#endif    // BINARYPOSITIONWRAPPER_SRC_DATASET_DATASET_H_
