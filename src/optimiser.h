
#ifndef DIFFERENTIATION_OPTIMISER_H
#define DIFFERENTIATION_OPTIMISER_H

#include "Data.h"
#include "Layer.h"
#include "math.h"

struct Adam{

private:
    Data**  first_moment_vector = nullptr;
    Data** second_moment_vector = nullptr;

    int time = 0;

    std::vector<LayerInterface*> layers;

public:
    double alpha = 0.001;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double eps   = 1e-8;


    const int count;

    Adam(std::vector<LayerInterface*> layers);

    void apply(Data* values, Data* gradient, Data* first_moment, Data* second_moment);

    void apply(ThreadData* td);

    virtual ~Adam();

};

#endif //DIFFERENTIATION_OPTIMISER_H
