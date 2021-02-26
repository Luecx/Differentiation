//
// Created by Luecx on 21.02.2021.
//

#ifndef DIFFERENTIATION_DENSELAYER_H
#define DIFFERENTIATION_DENSELAYER_H

#include "Function.h"
#include "math.h"
#include "Layer.h"
#include "config.h"
#include "matmul.h"


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
    void apply(Input *in, ThreadData* td){
        this->apply(in, td->output[layerID]);
    }
    void apply(Input *in, Data *out){
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


#endif //DIFFERENTIATION_DENSELAYER_H
