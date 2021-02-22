//
// Created by Luecx on 21.02.2021.
//

#ifndef DIFFERENTIATION_DENSELAYER_H
#define DIFFERENTIATION_DENSELAYER_H

#include "Data.h"
#include "Function.h"
#include "math.h"

static int layer_id_counter;

struct LayerInterface{
protected:
    int layerID;
public:
    virtual DataInterface* getBias   () = 0;
    virtual DataInterface* getWeights() = 0;

    virtual int getOutputSize() = 0;
    virtual int getInputSize()  = 0;

    virtual DataInterface* newOutputInstance() = 0;
    virtual DataInterface* newWeightInstance() = 0;
    virtual DataInterface* newBiasInstance() = 0;

    void assignID(){
        layerID = layer_id_counter ++;
    }
};


class ThreadData {
private:
    int count;



public:

    DataInterface** output;
    DataInterface** output_gradient;
    DataInterface** weight_gradient;
    DataInterface**   bias_gradient;

    ThreadData(std::vector<LayerInterface*> layers){
        count = layers.size();
        output              = new  DataInterface*[layers.size()];
        output_gradient     = new  DataInterface*[layers.size()];
        weight_gradient     = new  DataInterface*[layers.size()];
        bias_gradient       = new  DataInterface*[layers.size()];
        for(int i = 0; i < layers.size(); i++){
            output[i]           = layers[i]->newOutputInstance();
            output_gradient[i]  = layers[i]->newOutputInstance();
            weight_gradient[i]  = layers[i]->newWeightInstance();
            bias_gradient[i]    = layers[i]->newBiasInstance();
        }
    }

    virtual ~ThreadData() {
        for(int i = 0; i < count; i++){
            _mm_free(output[i]);
            _mm_free(output_gradient[i]);
            _mm_free(weight_gradient[i]);
            _mm_free(bias_gradient[i]);
        }

        _mm_free(output);
        _mm_free(output_gradient);
        _mm_free(weight_gradient);
        _mm_free(bias_gradient);

    }
};


template<int I, int O, template<int> class F>
class DenseLayer : public LayerInterface{
public:
    Data<O,I> weights                   {};
    Data<O  > bias                      {};
    F   <O  > f                         {};

    DenseLayer(Data<O,I> weights, Data<O> bias) {
        this->weights = weights;
        this->bias    = bias;
        this->assignID();
    }

    DenseLayer() {
        weights.randomise(-1.0 / sqrt(I), 1.0 / sqrt(I));
        bias   .randomise(-1.0 / sqrt(I), 1.0 / sqrt(I));
        this->assignID();
    }


    void apply(ThreadData* td){
       this->apply(
               dynamic_cast<Data<I>*>(td->output[layerID-1]),
               dynamic_cast<Data<O>*>(td->output[layerID]));
    }
    void apply(Data<I>  *in, Data<O> *out){
        matmul(&weights, in, out);
        out->add(&bias);
        f.apply(out, out);
    }
    void apply(Input    *in, ThreadData* td){
        this->apply(in, dynamic_cast<Data<O>*>(td->output[layerID]));
    }
    void apply(Input    *in, Data<O> *out){
        matmul(&weights, in, out);
        out->add(&bias);
        f.apply(out, out);
    }

    void backprop(
            ThreadData* td){
        this->backprop(
                dynamic_cast<Data<I  >*>(td->output[layerID-1]),
                dynamic_cast<Data<O  >*>(td->output[layerID]),
                dynamic_cast<Data<I  >*>(td->output_gradient[layerID-1]),
                dynamic_cast<Data<O  >*>(td->output_gradient[layerID]),
                dynamic_cast<Data<O,I>*>(td->weight_gradient[layerID]),
                dynamic_cast<Data<O  >*>(td->bias_gradient[layerID]));
    }
    void backprop(
            Data<I> *in,
            Data<O> *output,
            Data<I> *in_grad,
            Data<O> *out_grad,
            Data<O,I> *weights_grad,
            Data<O> *bias_grad){

        f.backprop(output, out_grad,out_grad);
        bias_grad->add(out_grad);
        matmul_backprop(&weights, in, weights_grad, in_grad, out_grad);
    }
    void backprop(
            Input *in,
            ThreadData* td){
        this->backprop(
                in,
                dynamic_cast<Data<O  >*>(td->output[layerID]),
                dynamic_cast<Data<O  >*>(td->output_gradient[layerID]),
                dynamic_cast<Data<O,I>*>(td->weight_gradient[layerID]),
                dynamic_cast<Data<O  >*>(td->bias_gradient[layerID]));
    }
    void backprop(
            Input     *in,
            Data<O>   *output,
            Data<O>   *out_grad,
            Data<O,I> *weights_grad,
            Data<O>   *bias_grad){
        f.backprop(output, out_grad,out_grad);
        bias_grad->add(out_grad);
        matmul_backprop(in, weights_grad, out_grad);
    }

    int getOutputSize() override {
        return O;
    }
    int getInputSize() override {
        return I;
    }

    DataInterface *getBias() override {
        return &bias;
    }
    DataInterface *getWeights() override {
        return &weights;
    }
    DataInterface *newOutputInstance() override {
        return new (std::align_val_t(32))  Data<O>{};
    }
    DataInterface *newWeightInstance() override {
        return new (std::align_val_t(32)) Data<O,I>{};
    }
    DataInterface *newBiasInstance() override {
        return new (std::align_val_t(32)) Data<O>{};
    }
};


template<int I, int O, template<int> class F>
class DuplicateDenseLayer : public LayerInterface{
public:
    Data<O,I> weights                   {};
    Data<O  > bias                      {};
    F   <O*2> f                         {};
    Data<O,I> weights_grad[N_THREADS]   {};
    Data<O  >    bias_grad[N_THREADS]   {};
    Data<O*2> output      [N_THREADS]   {};
    Data<O  > output_1    [N_THREADS]   {};
    Data<O  > output_2    [N_THREADS]   {};
    Data<O*2> output_grad [N_THREADS]   {};

    DuplicateDenseLayer(Data<O,I> weights, Data<O> bias) {
        this->weights = weights;
        this->bias    = bias;
    }

    DuplicateDenseLayer() {
        weights.randomise(-1.0 / sqrt(I), 1.0 / sqrt(I));
        bias   .randomise(-1.0 / sqrt(I), 1.0 / sqrt(I));
        weights.assignID();
        bias.assignID();
    }

    DataInterface *getBias() override {
        return &bias;
    }

    DataInterface *getWeights() override {
        return &weights;
    }

    void apply(Data<I>  *in, int thread_id=0){
        this->apply(in, &output[thread_id]);
    }
    void apply(Data<I>  *in, Data<O> *out){
        matmul(&weights, in, out);
        out->add(&bias);
        f.apply(out, out);
    }
    void apply(Input    *in, int thread_id=0){
        this->apply(in, &output[thread_id]);
    }
    void apply(Input    *in, Data<O> *out){
        matmul(&weights, in, out);
        out->add(&bias);
        f.apply(out, out);
    }

    void backprop(
            Data<I> *in,
            Data<I> *in_grad,
            int thread_id=0){
        this->backprop(
                in,
                &output[thread_id],
                in_grad,
                &output_grad[thread_id],
                &weights_grad[thread_id],
                &bias_grad[thread_id]);
    }
    void backprop(
            Data<I  > *in,
            Data<O  > *output,
            Data<I  > *in_grad,
            Data<O  > *out_grad,
            Data<O,I> *weights_grad,
            Data<O  > *bias_grad){

        f.backprop(output, out_grad,out_grad);
        bias_grad->add(out_grad);
        matmul_backprop(&weights, in, weights_grad, in_grad, out_grad);
    }
    void backprop(
            Input *in,
            int thread_id=0){
        this->backprop(
                in,
                &output[thread_id],
                &output_grad[thread_id],
                &weights_grad[thread_id],
                &bias_grad[thread_id]);
    }
    void backprop(
            Input     *in,
            Data<O  > *output,
            Data<O  > *out_grad,
            Data<O,I> *weights_grad,
            Data<O  > *bias_grad){
        f.backprop(output, out_grad,out_grad);
        bias_grad->add(out_grad);
        matmul_backprop(in, weights_grad, out_grad);
    }
};

#endif //DIFFERENTIATION_DENSELAYER_H
