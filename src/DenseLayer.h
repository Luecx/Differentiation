//
// Created by Luecx on 21.02.2021.
//

#ifndef DIFFERENTIATION_DENSELAYER_H
#define DIFFERENTIATION_DENSELAYER_H

#include "Function.h"
#include "math.h"
#include "Layer.h"



template<int I, int O, typename F>
class DenseLayer : public LayerInterface{
public:
    Data weights                   {O,I};
    Data bias                      {O};
    F    f                         {};

    DenseLayer() {
        weights.randomise(-1.0 / sqrt(I), 1.0 / sqrt(I));
        bias   .randomise(-1.0 / sqrt(I), 1.0 / sqrt(I));
        assignID();
    }

    void apply(ThreadData* td){
       this->apply(
               td->output[layerID-1],
               td->output[layerID]);
    }
    void apply(Data  *in, Data *out){
        matmul(&weights, in, out);
        out->add(&bias);
        f.apply(out, out);
    }
    void apply(Input    *in, ThreadData* td){
        this->apply(in, td->output[layerID]);
    }
    void apply(Input    *in, Data *out){
        matmul(&weights, in, out);
        out->add(&bias);
        f.apply(out, out);
    }

    void backprop(
            ThreadData* td){
        this->backprop(
                td->output[layerID-1],
                td->output[layerID],
                td->output_gradient[layerID-1],
                td->output_gradient[layerID],
                td->weight_gradient[layerID],
                td->bias_gradient[layerID]);
    }

    void backprop(
            Data *in,
            Data *output,
            Data *in_grad,
            Data *out_grad,
            Data *weights_grad,
            Data *bias_grad){

        f.backprop(output, out_grad,out_grad);
        bias_grad->add(out_grad);
        matmul_backprop(&weights, in, weights_grad, in_grad, out_grad);
    }

    void backprop(
            Input *in,
            ThreadData* td){
        this->backprop(
                in,
                td->output[layerID],
                td->output_gradient[layerID],
                td->weight_gradient[layerID],
                td->bias_gradient[layerID]);
    }

    void backprop(
            Input     *in,
            Data   *output,
            Data   *out_grad,
            Data   *weights_grad,
            Data   *bias_grad){
        f.backprop(output, out_grad,out_grad);
        bias_grad->add(out_grad);
        matmul_backprop(in, weights_grad, out_grad);
    }

    void assignThreadData(ThreadData **td) override {

    }
    int  getOutputSize() override {
        return O;
    }
    int  getInputSize() override {
        return I;
    }

    Data *getBias() override {
        return &bias;
    }
    Data *getWeights() override {
        return &weights;
    }
    Data *newOutputInstance() override {
        return new Data(O);
    }
    Data *newWeightInstance() override {
        return new Data(O,I);
    }
    Data *newBiasInstance() override {
        return new Data(O);
    }
};


template<int I, int O, typename F>
class DuplicateDenseLayer : public LayerInterface{
public:
    Data      weights                   {O,I};
    Data      bias                      {O};
    F         f                         {};
    Data*     im1[N_THREADS]            {};
    Data*     im2[N_THREADS]            {};
    Data*     im1_g[N_THREADS]          {};
    Data*     im2_g[N_THREADS]          {};

    DuplicateDenseLayer() {
        weights.randomise(-1.0 / sqrt(I), 1.0 / sqrt(I));
        bias   .randomise(-1.0 / sqrt(I), 1.0 / sqrt(I));
        this->assignID();
    }



    void apply(Input    *in1,
               Input    *in2,
               ThreadData* td){
        matmul(&weights, in1, im1[td->threadID]);
        matmul(&weights, in2, im2[td->threadID]);
        im1[td->threadID]->add(&bias);
        im2[td->threadID]->add(&bias);
        f.apply(im1[td->threadID], im1[td->threadID]);
        f.apply(im2[td->threadID], im2[td->threadID]);
    }

    void backprop(
            Input      *in1,
            Input      *in2,
            ThreadData *td){
        f.backprop(im1[td->threadID], im1_g[td->threadID],im1_g[td->threadID]);
        f.backprop(im2[td->threadID], im2_g[td->threadID],im2_g[td->threadID]);
        td->bias_gradient[layerID] = im1_g[td->threadID];
        td->bias_gradient[layerID]->add(im2_g[td->threadID]);

        matmul_backprop(in1, td->weight_gradient[layerID], im1_g[td->threadID]);
        matmul_backprop(in2, td->weight_gradient[layerID], im2_g[td->threadID]);

    }


    void assignThreadData(ThreadData** td){
        for(int i = 0; i < N_THREADS; i++){
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
        return I;
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
};

#endif //DIFFERENTIATION_DENSELAYER_H
