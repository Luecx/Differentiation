//
// Created by Luecx on 24.02.2021.
//

#ifndef DIFFERENTIATION_CONFIG_H
#define DIFFERENTIATION_CONFIG_H


// change this to scale the input to the sigmoid function
#define SIGMOID_SCALE (2.5 / 400.0)

// change this to adjust the amount of threads the neural network will use
#define NN_THREADS (8)

// change this to   ds adjust the amount of thread will be used for merging gradients and adjusting weights.
#define UPDATE_THREADS (12)

// note that more NN_THREADS leads to a large overhead for merging gradients.
// Especially with smaller batch sizes, the amount of NN_THREADS should be smaller than the UPDATE_THREADS since
// merging will take most of the time.

#endif //DIFFERENTIATION_CONFIG_H
