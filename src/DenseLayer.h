//
// Created by Luecx on 21.02.2021.
//

#ifndef DIFFERENTIATION_DENSELAYER_H
#define DIFFERENTIATION_DENSELAYER_H

#include "Function.h"
#include <cmath>
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
//        weights.randomise(-1.0 / sqrt(I), 1.0 / sqrt(I));
//        bias   .randomise(-1.0 / sqrt(I), 1.0 / sqrt(I));

//

        weights.randomiseGaussian(0, sqrt(2.0 / I));
//        bias   .randomiseGaussian(0, 0);
//        weights.randomiseKieren();
        bias.randomiseGaussian(0,0);

//        weights.randomise(0, 1.0 / sqrt(I));
//        bias   .randomise(0, 1.0 / sqrt(I));
    }

    void apply(ThreadData* td) override{
       this->apply(
               td->output[layerID-1],
               td->output[layerID]);
    }
    void apply(Data  *in, Data *out){
        matmul(&weights, in, out);
        out->add(&bias);
        f.apply(out, out);
    }
    void apply(Input *in, ThreadData* td) override{
        this->apply(in, td->output[layerID]);
    }
    void apply(Input *in, Data *out){
        matmul(&weights, in, out);
        out->add(&bias);
        f.apply(out, out);
    }
    void apply(Data *input, ThreadData *td) override {
        this->apply(input, td->output[layerID]);
    }


    void backprop(
            ThreadData* td) override{
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
            ThreadData* td) override{
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

    void backprop(
        Data   *in,
        Data   *output,
        Data   *out_grad,
        Data   *weights_grad,
        Data   *bias_grad){
        f.backprop(output, out_grad,out_grad);
        bias_grad->add(out_grad);
        matmul_backprop(&weights, in, weights_grad, out_grad);
    }

    void backprop(Data *input, ThreadData *td) override {
        this->backprop(
            input,
            td->output[layerID],
            td->output_gradient[layerID],
            td->weight_gradient[layerID],
            td->bias_gradient[layerID]);
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
    Activation* getActivationFunction() override { return &f; }
};


#endif //DIFFERENTIATION_DENSELAYER_H
