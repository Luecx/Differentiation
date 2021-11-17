
#ifndef DIFFERENTIATION_SRC_WEIGHTCLAMP_H_
#define DIFFERENTIATION_SRC_WEIGHTCLAMP_H_

#include "../structures/Data.h"
class WeightClamp {
    float min_clip;
    float max_clip;

    public:
    explicit WeightClamp(float min_clip=-__FLT_MAX__, float max_clip=__FLT_MAX__);

    void apply(Data* data) const;

};

#endif    // DIFFERENTIATION_SRC_WEIGHTCLAMP_H_
