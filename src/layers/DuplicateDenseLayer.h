//
// Created by Luecx on 25.02.2021.
//

#ifndef DIFFERENTIATION_DUPLICATEDENSELAYER_H
#define DIFFERENTIATION_DUPLICATEDENSELAYER_H


template<int I, int O, typename F>
class DuplicateDenseLayer : public LayerInterface{
public:
    Data      weights                    {O,I};
    Data      bias                       {O};
    F         f                          {};
    Data*     im1[NN_THREADS]            {};
    Data*     im2[NN_THREADS]            {};
    Data*     im1_g[NN_THREADS]          {};
    Data*     im2_g[NN_THREADS]          {};

    DuplicateDenseLayer(int expected_active_inputs=I) {
        weights.randomiseGaussian(0, 2.0 / sqrt(expected_active_inputs));
        bias.randomiseGaussian   (0,0);
    }


    void apply(ThreadData *td) override {

    }
    void backprop(ThreadData *td) override {

    }

    void apply(Data *input, ThreadData *td) override {

    }
    void backprop(Data *input, ThreadData *td) override {

    }

    void apply(Input    *in1,
               ThreadData* td){
        matmul(&weights, in1, im1[td->threadID], 0);
        matmul(&weights, in1, im2[td->threadID], I);
        im1[td->threadID]->add(&bias);
        im2[td->threadID]->add(&bias);
        f.apply(im1[td->threadID], im1[td->threadID]);
        f.apply(im2[td->threadID], im2[td->threadID]);
    }

    void backprop(
            Input      *in1,
            ThreadData *td){
        f.backprop(im1[td->threadID], im1_g[td->threadID],im1_g[td->threadID]);
        f.backprop(im2[td->threadID], im2_g[td->threadID],im2_g[td->threadID]);

        *td->bias_gradient[layerID] =   *im1_g[td->threadID];
         td->bias_gradient[layerID]->add(im2_g[td->threadID]);

        matmul_backprop(in1, td->weight_gradient[layerID], im1_g[td->threadID], 0);
        matmul_backprop(in1, td->weight_gradient[layerID], im2_g[td->threadID], I);
    }


    void assignThreadData(ThreadData** td) override{
        for(int i = 0; i < NN_THREADS; i++){
            Data* out   = td[i]->output           [layerID];
            Data* out_g = td[i]->output_gradient  [layerID];
            im1  [i] = new Data(&(out  ->values[0]), O);
            im2  [i] = new Data(&(out  ->values[O]), O);
            im1_g[i] = new Data(&(out_g->values[0]), O);
            im2_g[i] = new Data(&(out_g->values[O]), O);
        }
    }
    int getOutputSize() override {
        return O*2;
    }
    int getInputSize() override {
        return I*2;
    }
    Data *newOutputInstance() override {
        return new Data(O*2);
    }
    Data *newWeightInstance() override {
        return new Data(O, I);
    }
    Data *newBiasInstance() override {
        return new Data(O);
    }
    Data *getBias() override {
        return &bias;
    }
    Data *getWeights() override {
        return &weights;
    }
    Activation* getActivationFunction() override { return &f; }
};

#endif //DIFFERENTIATION_DUPLICATEDENSELAYER_H
