//
// Created by Luecx on 21.02.2021.
//

#ifndef DIFFERENTIATION_DATA_H
#define DIFFERENTIATION_DATA_H

#include <immintrin.h>
#include <ostream>
#include <iomanip>
#include <iostream>
#include <cstring>
#include "Input.h"
#include "util.h"

#define ALIGNMENT 64
#define N_THREADS 32
#define PARALLEL_SIZE_32_BIT(x) x - x %  8

typedef int Dimension;



struct DataInterface{
    virtual int getM() = 0;
    virtual int getN() = 0;

    virtual DataInterface* newInstance() = 0;


};

template<Dimension M, Dimension N=1>
class Data : public DataInterface{

public:

    alignas(ALIGNMENT) float values[M*N];

    float  get(int height)                   const { return values[height]; }
    float& get(int height)                         { return values[height]; }
    float  get(int height, int width)        const { return values[width + height * N]; }
    float& get(int height, int width)              { return values[width + height * N]; }
    float  operator()(int height)            const { return get(height); }
    float& operator()(int height)                  { return get(height); }
    float  operator()(int height, int width) const { return get(height, width); }
    float& operator()(int height, int width)       { return get(height, width); }

    int getM() override {
        return M;
    }
    int getN() override {
        return N;
    }

    Data<M,N>& operator=(const Data<M,N> &other){
        std::memcpy(values, other.values, sizeof(float) * M * N);
        return *this;
    }

    void   clear(){
        std::memset(values, 0, sizeof(float) * M * N);
    }
    void   randomise(float lower, float upper) {
        for (int i = 0; i < M*N; i++) {
            this->values[i] = static_cast<float>(rand()) / RAND_MAX * (upper - lower) + lower;
        }
    }
    void   add(Data<M,N> *other){
        constexpr int size = PARALLEL_SIZE_32_BIT(M*N);
        for (int i = 0; i < size; i += 8) {
            // load our values and the target values into the register
            __m256* other_values = (__m256 *)(&other->values[i]);
            __m256* our_values   = (__m256 *)( &this->values[i]);
            // stores the sum of our and their values inside the other data object.
            *our_values = _mm256_add_ps(*other_values, *our_values);
        }
        for (int i = size; i < M*N; i ++) {
            (*this)(i) += (*other)(i);
        }
    }
    void   add(Data<M,N> *other, float scalar){
        constexpr int size = PARALLEL_SIZE_32_BIT(M*N);
        __m256 s = _mm256_set1_ps(scalar);
        for (int i = 0; i < size; i += 8) {
            // load our values and the target values into the register
            __m256* other_values = (__m256 *)(&other->values[i]);
            __m256* our_values   = (__m256 *)( &this->values[i]);
            // stores the sum of our and their values inside the other data object.
            *our_values = _mm256_add_ps(_mm256_mul_ps(s,*other_values), *our_values);
        }
        for (int i = size; i < M*N; i ++) {
            (*this)(i) += (*other)(i) * scalar;
        }
    }
    void   sub(Data<M,N> *other){
        constexpr int size = PARALLEL_SIZE_32_BIT(M*N);
        for (int i = 0; i < size; i += 8) {
            // load our values and the target values into the register
            __m256* other_values = (__m256 *)(&other->values[i]);
            __m256* our_values   = (__m256 *)( &this->values[i]);
            // stores the sum of our and their values inside the other data object.
            *our_values = _mm256_sub_ps(*our_values, *other_values);
        }
        for (int i = size; i < M*N; i ++) {
            (*this)(i) -= (*other)(i);
        }
    }
    DataInterface *newInstance() override {
        return new Data<M,N>{};
    }

    friend std::ostream& operator<<(std::ostream& os, const Data<M,N>& data) {

        if(N != 1){
            os << std::fixed << std::setprecision(3);
            for (int i = 0; i < M; i++) {
                for (int n = 0; n < N; n++) {
                    os << std::setw(11) << (double)data(i,n);
                }
                os << "\n";
            }
        }else{
            os << "(transposed) ";
            for (int n = 0; n < M; n++) {
                os << std::setw(11) << (double)data(n);
            }
        }
        return os;

    }

};

template<Dimension M, Dimension N>
inline void matmul(
        Data<M,N>* weights,
        Data<N,1>* vector,
        Data<M,1>* target){

    static_assert(N % 8 == 0);

    __m256* wgt       = (__m256*)(weights->values);
    __m256* inp       = (__m256*)(vector ->values);
    __m256* output    = (__m256*)(target ->values);


    constexpr int    row_chunks = M / 8;
    constexpr int column_chunks = N / 8;


    for (int row_chunk = 0; row_chunk < row_chunks; row_chunk++) {

        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();
        for (int col_chunk = 0; col_chunk < column_chunks; col_chunk++) {
            __m256 vec = inp[col_chunk];

            // other->height * row_chunk + col_chunk is constant and wrapped into offset
            const int offset = N * row_chunk + col_chunk;

            // computation of entry:
            // other->height * row_chunk = beginning of the rows
            // column_chunks * n         = skipping to the specified row n
            // col_chunk                 = skipping to the current column
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(wgt[offset + column_chunks * 0], vec));
            acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(wgt[offset + column_chunks * 1], vec));
            acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(wgt[offset + column_chunks * 2], vec));
            acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(wgt[offset + column_chunks * 3], vec));
            acc4 = _mm256_add_ps(acc4, _mm256_mul_ps(wgt[offset + column_chunks * 4], vec));
            acc5 = _mm256_add_ps(acc5, _mm256_mul_ps(wgt[offset + column_chunks * 5], vec));
            acc6 = _mm256_add_ps(acc6, _mm256_mul_ps(wgt[offset + column_chunks * 6], vec));
            acc7 = _mm256_add_ps(acc7, _mm256_mul_ps(wgt[offset + column_chunks * 7], vec));
        }

        acc0 = _mm256_hadd_ps(acc0, acc1);
        acc2 = _mm256_hadd_ps(acc2, acc3);

        acc4 = _mm256_hadd_ps(acc4, acc5);
        acc6 = _mm256_hadd_ps(acc6, acc7);

        acc0 = _mm256_hadd_ps(acc0, acc2);
        acc4 = _mm256_hadd_ps(acc4, acc6);

        __m128 sumabcd1 = _mm256_extractf128_ps(acc0, 0);
        __m128 sumabcd2 = _mm256_extractf128_ps(acc0, 1);
        __m128 sumefgh1 = _mm256_extractf128_ps(acc4, 0);
        __m128 sumefgh2 = _mm256_extractf128_ps(acc4, 1);

        sumabcd1 = _mm_add_ps(sumabcd1, sumabcd2);
        sumefgh1 = _mm_add_ps(sumefgh1, sumefgh2);

        acc0 = _mm256_insertf128_ps(_mm256_castps128_ps256(sumabcd1), sumefgh1, 1);
        output[row_chunk] = acc0;
    }

    // doing the remaining rows
    // no need for this to be efficient as this will only be called rarely
    for (int row = row_chunks * 8; row < M; row++){
        __m256 acc0 = _mm256_setzero_ps();
        for (int col = 0; col < N; col += 8) {
            __m256 vec = _mm256_load_ps(&vector->values[col]);
            __m256 mat0 = _mm256_load_ps(&weights->values[col + N * row]);
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(mat0, vec));
        }
        target->values[row] = acc0[0] + acc0[1] + acc0[2] + acc0[3] +
                              acc0[4] + acc0[5] + acc0[6] + acc0[7];
    }

    return;

}

template<Dimension M, Dimension N>
inline void matmul_backprop(
        Data<M,N>* weights,
        Data<N,1>* vector,
        Data<M,N>* weights_grad,
        Data<N,1>* vector_grad,
        Data<M,1>* target_grad){

    static_assert(N % 8 == 0);

    vector_grad->clear();

    __m256* wgt             = (__m256*)(weights           ->values);
    __m256* wgt_grd         = (__m256*)(weights_grad      ->values);
    __m256* inp             = (__m256*)(vector            ->values);
    __m256* inp_grd         = (__m256*)(vector_grad       ->values);

    constexpr int    row_chunks = M;
    constexpr int column_chunks = N / 8;

    for(int row_chunk = 0; row_chunk < row_chunks; row_chunk ++){
        __m256 k1   = _mm256_set1_ps(target_grad->get(row_chunk));
        for(int col_chunk = 0; col_chunk < column_chunks; col_chunk ++){

            int weight_index = row_chunk * column_chunks + col_chunk;
            inp_grd[col_chunk]    = _mm256_add_ps(inp_grd [col_chunk   ], _mm256_mul_ps(wgt [weight_index], k1));
            wgt_grd[weight_index] = _mm256_add_ps(wgt_grd [weight_index], _mm256_mul_ps(inp [col_chunk   ], k1));
        }
    }
}


template<Dimension M, Dimension N>
inline void matmul(
        Data<M,N>* weights,
        Input    * vector,
        Data<M,1>* target){

    target->clear();


    // extract the output values to which we write the transformation
    // there is no input data object as the Sample contains all relevant information
    // Note that the sample only contains indices for the places where there is a "1"
    // -> there is only 0 or 1 as an input for a neuron
    // The transformation is similar to the affine transformation which applies: o = A*x + b
    // where A is the weights matrix, b is the bias and x is the input encoded in the sample.
    float* outputValues = target ->values;
    float* weightValues = weights->values;

    for(uint16_t &index:vector->indices){

        // we can only do the chunks of 8 with avx instructions
        // the rest must be done manually
        int size = PARALLEL_SIZE_32_BIT(M);
        // we assume that the output size of the very first layer is always a multiple of 8!
        for(int n = 0; n < size; n+=8){
            // get the gradients into the register aswell as the output which we want to write to
            __m256 wvalues = _mm256_load_ps(&(weightValues[index * M + n]));
            __m256 ovalues = _mm256_load_ps(&(outputValues[            n]));
            // add the element-wise multiplication of the weights. For this, add the weights for the activated
            // input neuron (output = 1) to the output
            _mm256_store_ps(&outputValues[n],_mm256_add_ps(ovalues, wvalues));
        }
        for(int n = size; n < M; n++){
            outputValues[n] += weightValues[index * M + n];
        }
    }

}


template<Dimension M, Dimension N>
inline void matmul_backprop(
        Input    * vector,
        Data<M,N>* weights_grad,
        Data<M,1>* target_grad){

    // extract the weight gradient values which we want to compute.
    // there is no input data object as the Sample contains all relevant information.
    // Note that the sample only contains indices for the places where there is a "1"
    // -> there is only 0 or 1 as an input for a neuron
    // The transformation is similar to the backpropagation of the affine transformation
    // which computes gradients for a weights connecting node i with node o by doing:
    // grad(w_io) += output(i) * grad(o)
    float* weightsGrad = weights_grad->values;
    float* outputGrad  =  target_grad->values;

    // going through each index, applying the rules described above
    // Note that this assumes, as well as the forward step, that the output size is a multiple of 8
    // Otherwise a SIGSEGV will occur as we try to load 256 bit into a register to which we dont have access.
    for(uint16_t &index:vector->indices){
        // we can only do the chunks of 8 with avx instructions
        // the rest must be done manually
        int size = PARALLEL_SIZE_32_BIT(M);
        for(int n = 0; n < size; n+=8){
            // get the weight gradient which we want to increment as well as the output gradient
            __m256 wgrad = _mm256_load_ps(&(weightsGrad[index * M + n]));
            __m256 ograd = _mm256_load_ps(&( outputGrad[            n]));

            _mm256_store_ps(&(weightsGrad[index * M + n]), _mm256_add_ps(wgrad, ograd));
        }
        for(int n = size; n < M; n++){
            weightsGrad[index * M + n] += outputGrad[n];
        }
    }

}



#endif //DIFFERENTIATION_DATA_H
