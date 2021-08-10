//
// Created by Luecx on 25.02.2021.
//
#include "merge.h"

void merge_gradients(ThreadData** td) {
    if (NN_THREADS == 1)
        return;

    for (int l = 0; l < td[0]->count; l++) {
        const int chunk_size = std::max(1, td[0]->weight_gradient[l]->size() / UPDATE_THREADS);
#pragma omp parallel for schedule(static, chunk_size) num_threads(UPDATE_THREADS)
        for (int s = 0; s < td[0]->weight_gradient[l]->size(); s++) {
            for (int t = 1; t < NN_THREADS; t++) {
                td[0]->weight_gradient[l]->values[s] += td[t]->weight_gradient[l]->values[s];
                td[t]->weight_gradient[l]->values[s] = 0;
            }
        }

        for (int s = 0; s < td[0]->bias_gradient[l]->size(); s++) {
            for (int t = 1; t < NN_THREADS; t++) {
                td[0]->bias_gradient[l]->values[s] += td[t]->bias_gradient[l]->values[s];
                td[t]->bias_gradient[l]->values[s] = 0;
            }
        }
    }
}
