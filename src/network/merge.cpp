//
// Created by Luecx on 25.02.2021.
//

#include "merge.h"
#include "ThreadData.h"

void merge_gradients(ThreadData** td, Data* activated_inputs) {

    // no merging required incase of a single thread
    if (NN_THREADS == 1)
        return;

    // check if there is a map for the input weights which can be used
    if(activated_inputs != nullptr){

        // cols and rows are reversed here, since the indexing inside matmul(Input*... is also reversed)
        int cols = td[0]->weight_gradient[0]->M;
        int rows = td[0]->weight_gradient[0]->N;

        // compute how much each thread should do
        const int chunk_size = std::max(1, rows / UPDATE_THREADS);

        // run over the rows. note that we cannot use the get()
        // function since the matrix operation reverses the indexing in order
        // to allow sparse inputs
#pragma omp parallel for schedule(static, chunk_size) num_threads(UPDATE_THREADS)
        for(int c = 0; c < cols; c++){
            // dont do this column if the input was never used
            if(activated_inputs->get(c) == 0) continue;
            // add together
            for (int r = 0; r < rows; r++) {
                int index = c * rows + r;
                for (int t = 1; t < NN_THREADS; t++) {
                    td[0]->weight_gradient[0]->values[index] += td[t]->weight_gradient[0]->values[index];
                    td[t]->weight_gradient[0]->values[index]  = 0;
                }
            }
        }

        // do biases
        for (int s = 0; s < td[0]->bias_gradient[0]->size(); s++) {
            for (int t = 1; t < NN_THREADS; t++) {
                td[0]->bias_gradient[0]->values[s] += td[t]->bias_gradient[0]->values[s];
                td[t]->bias_gradient[0]->values[s] = 0;
            }
        }
    }

    // do the remaining layers
    // check if we have used some input map before. if so, we can skip the first layer
    for (int l = (activated_inputs == nullptr ? 0 : 1); l < td[0]->count; l++) {
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
