
#ifndef DIFFERENTIATION_OPTIMISER_H
#define DIFFERENTIATION_OPTIMISER_H

#include "../layers/Layer.h"
#include "../structures/Data.h"

#include <cmath>

/**
 * basic interface for all optimisers.
 * An optimiser has to do the following:
 *
 * Use the gradients to optimise and adjust the values.
 */
struct Optimiser{
    /**
     * Perhaps it needs to create local data-buffers. For this, it gets a list of layers
     * which can be used to create copies of the weight-matrices and bias-vectors.
     * @param layers        the layers which the network consists of
     */
    virtual void init(std::vector<LayerInterface*> layers) = 0;

    /**
     * The function is supposed to go through the gradients of the weights and bias which are stored within
     * the ThreadData object.
     * Furthermore the gradients must be cleared in the process!
     * Beside the gradients, an adjustment of the learning rate can be done based on the amount of batches which
     * have had impact on the gradients.
     * @param td            ThreadData which contains the gradients
     * @param batch_size    batch size which has been used
     */
    virtual void apply(ThreadData* td, int batch_size) = 0;

    /**
     * If the optimiser requires information about when a new epoch starts, this function can be used.
     * This is mostly unused.
     */
    virtual void newEpoch() = 0;

    /**
     * Used to display information to the log file
     */
    virtual void logOverview() = 0;
};


#endif //DIFFERENTIATION_OPTIMISER_H
