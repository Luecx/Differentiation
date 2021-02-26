
#ifndef DIFFERENTIATION_OPTIMISER_H
#define DIFFERENTIATION_OPTIMISER_H

#include "Data.h"
#include "Layer.h"
#include "math.h"

struct Optimiser{

    virtual void init(std::vector<LayerInterface*> layers) = 0;

    virtual void apply(ThreadData* td, int batch_size) = 0;

    virtual void newEpoch() = 0;
};

struct Adam : Optimiser{

private:
    Data**  first_moment_vector = nullptr;
    Data** second_moment_vector = nullptr;

    int time = 1;
    int count = 0;

    std::vector<LayerInterface*> layers;

public:
    double alpha = 0.0001;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps   = 1e-8;



    void init(std::vector<LayerInterface*> layers);

    void apply(Data* values, Data* gradient, Data* first_moment, Data* second_moment);

    void apply(ThreadData* td, int batch_size);

    void newEpoch() override;

    virtual ~Adam();

};

#endif //DIFFERENTIATION_OPTIMISER_H
