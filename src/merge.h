//
// Created by Luecx on 23.02.2021.
//

#ifndef DIFFERENTIATION_MERGE_H
#define DIFFERENTIATION_MERGE_H

#include "Layer.h"


void merge_gradients(ThreadData** td){
    if(N_THREADS == 1) return;

    constexpr int chunks = sqrt(N_THREADS);
#pragma omp parallel for schedule(auto) num_threads(N_THREADS-1)
    for(int t = 1; t < N_THREADS; t++){
        for(int l = 0; l < td[0]->count; l++){
            td[0]->weight_gradient[l]->add(td[t]->weight_gradient[l]);
        }
    }
}

#endif //DIFFERENTIATION_MERGE_H
