
//
// Created by Luecx on 10.11.2021.
//

#ifndef DIFFERENTIATION_SRC_OPTIMISER_ADAM_H_
#define DIFFERENTIATION_SRC_OPTIMISER_ADAM_H_

#include "../layers/Layer.h"
#include "../network/ThreadData.h"
#include "optimiser.h"

/**
 * Adam is an optimiser which can be used for stochastic training.
 */
struct Adam : Optimiser{

    private:
    Data**  first_moment_vector = nullptr;
    Data** second_moment_vector = nullptr;

    int time = 1;
    int count = 0;

    std::vector<LayerInterface*> layers;

    public:
    double alpha = 0.01;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps   = 1e-8;


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
    void apply(Data* values, Data* gradient, Data* first_moment, Data* second_moment);

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

    void   logOverview() override;

    /**
     * Destructor to deallocate the moment estimates.
     */
    virtual ~Adam();

};


#endif    // DIFFERENTIATION_SRC_OPTIMISER_ADAM_H_
