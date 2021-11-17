//
// Created by Luecx on 23.02.2021.
//

#ifndef DIFFERENTIATION_MERGE_H
#define DIFFERENTIATION_MERGE_H

#include "../layers/Layer.h"

/**
 * Used to merge multiple gradients.
 * It takes in an array of ThreadData pointers.
 * For each layer, the gradients for both, the weights and the bias
 * will be collapsed into the first entry (threadData[0]).
 * Gradients for the remaining will also be cleared (set to 0) in the process.
 *
 * @param td The threadData's to be collapsed and cleared
 */
void merge_gradients(ThreadData **td, Data* activated_inputs= nullptr);

#endif //DIFFERENTIATION_MERGE_H
