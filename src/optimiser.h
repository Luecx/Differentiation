
#ifndef DIFFERENTIATION_OPTIMISER_H
#define DIFFERENTIATION_OPTIMISER_H

#include "Data.h"
#include "Layer.h"
#include "math.h"

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
};

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
    double alpha = 0.0001;
    double beta1 = 0.95;
    double beta2 = 0.999;
    double eps   = 1e-8;


    /**
     * Function to init the first moment and second moment vectors
     * @param layers        the layers which the network consists of
     */
    void init(std::vector<LayerInterface*> layers);

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
    void apply(ThreadData* td, int batch_size);

    /**
     * Unused
     */
    void newEpoch() override;

    /**
     * Destructor to deallocate the moment estimates.
     */
    virtual ~Adam();

};

/**
 * Gradient descent is an optimiser which can be used for stochastic training.
 */
struct Gd : Optimiser{

    private:

    int count = 0;

    std::vector<LayerInterface*> layers;

    public:
    double alpha = 0.0000005;

    /**
     * Function to init the first moment and second moment vectors
     * @param layers        the layers which the network consists of
     */
    void init(std::vector<LayerInterface*> layers);

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
    void apply(ThreadData* td, int batch_size);

    /**
     * Unused
     */
    void newEpoch() override;

    /**
     * Destructor to deallocate the moment estimates.
     */
    virtual ~Gd();

};

#endif //DIFFERENTIATION_OPTIMISER_H
