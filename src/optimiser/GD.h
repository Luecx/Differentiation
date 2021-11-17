
//
// Created by Luecx on 17.11.2021.
//

#ifndef DIFFERENTIATION_SRC_OPTIMISER_GD_H_
#define DIFFERENTIATION_SRC_OPTIMISER_GD_H_

#include "optimiser.h"


/**
 * Gradient descent is an optimiser which can be used for stochastic training.
 */
struct Gd : Optimiser{

    private:

    int count = 0;

    std::vector<LayerInterface*> layers;

    public:
    double alpha = 0.0001;

    /**
     * Function to init the first moment and second moment vectors
     * @param layers        the layers which the network consists of
     */
    void init(std::vector<LayerInterface*> layers) override;

    /**
     * applies adam to the values based on the gradients for the values.
     *
     * @param values            values to be adjusted
     * @param gradient          gradients for the values
     * @param first_moment      first moment estimates of the gradients
     * @param second_moment     second moment estimates of the gradients
     */
    void apply(Data* values, Data* gradient);

    /**
     * applies adam to the values based on the gradients for the values.
     *
     * @param td                the thread data object which contains the gradients
     * @param batch_size        the batch size which has been used for the batch
     */
    void apply(ThreadData* td, int batch_size) override;

    /**
     * Unused
     */
    void newEpoch() override;

    /**
     * Destructor to deallocate the moment estimates.
     */
    virtual ~Gd();

    void logOverview() override;
};


#endif    // DIFFERENTIATION_SRC_OPTIMISER_GD_H_
