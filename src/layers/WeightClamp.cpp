

#include "WeightClamp.h"
WeightClamp::WeightClamp(float min_clip, float max_clip) : min_clip(min_clip), max_clip(max_clip) {}

void WeightClamp::apply(Data* data) const {
    data->clamp(min_clip, max_clip);
}
