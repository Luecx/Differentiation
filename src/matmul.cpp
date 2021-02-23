//
// Created by Luecx on 23.02.2021.
//

#include "Data.h"

void matmul(
        const Data *weights,
        const Data *vector,
        Data *target) {

    assert(vector->M == weights->N);
    assert(target->M == weights->M);
    assert(weights->N % 8 == 0);


    __m256* wgt       = (__m256*)(weights->values);
    __m256* inp       = (__m256*)(vector ->values);
    __m256* output    = (__m256*)(target ->values);


    const int    row_chunks = weights->M / 8;
    const int column_chunks = weights->N / 8;


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
            const int offset = weights->N * row_chunk + col_chunk;

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
    for (int row = row_chunks * 8; row < weights->M; row++){
        __m256 acc0 = _mm256_setzero_ps();
        for (int col = 0; col < weights->N; col += 8) {
            __m256 vec = _mm256_load_ps(&vector->values[col]);
            __m256 mat0 = _mm256_load_ps(&weights->values[col + weights->N * row]);
            acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(mat0, vec));
        }
        target->values[row] = acc0[0] + acc0[1] + acc0[2] + acc0[3] +
                              acc0[4] + acc0[5] + acc0[6] + acc0[7];
    }

    return;

}

void matmul_backprop(
        const Data *weights,
        const Data *vector,
        Data *weights_grad,
        Data *vector_grad,
        const Data *target_grad) {

    assert(vector_grad->M == vector->M);
    assert(weights_grad->M == weights->M);
    assert(weights_grad->N == weights->N);
    assert(vector->M == weights->N);
    assert(target_grad->M == weights->M);
    assert(weights->N % 8 == 0);


    vector_grad->clear();

    __m256* wgt             = (__m256*)(weights           ->values);
    __m256* wgt_grd         = (__m256*)(weights_grad      ->values);
    __m256* inp             = (__m256*)(vector            ->values);
    __m256* inp_grd         = (__m256*)(vector_grad       ->values);

    const int    row_chunks = weights->M;
    const int column_chunks = weights->N / 8;

    for(int row_chunk = 0; row_chunk < row_chunks; row_chunk ++){
        __m256 k1   = _mm256_set1_ps(target_grad->get(row_chunk));
        for(int col_chunk = 0; col_chunk < column_chunks; col_chunk ++){

            int weight_index = row_chunk * column_chunks + col_chunk;
            inp_grd[col_chunk]    = _mm256_add_ps(inp_grd [col_chunk   ], _mm256_mul_ps(wgt [weight_index], k1));
            wgt_grd[weight_index] = _mm256_add_ps(wgt_grd [weight_index], _mm256_mul_ps(inp [col_chunk   ], k1));
        }
    }
}

void matmul(
        const Data *weights,
        const Input *vector,
        Data *target) {

    target->clear();

    assert(target->M == weights->M);
    assert(weights->N % 8 == 0);


    // extract the output values to which we write the transformation
    // there is no input data object as the Sample contains all relevant information
    // Note that the sample only contains indices for the places where there is a "1"
    // -> there is only 0 or 1 as an input for a neuron
    // The transformation is similar to the affine transformation which applies: o = A*x + b
    // where A is the weights matrix, b is the bias and x is the input encoded in the sample.
    float* outputValues = target ->values;
    float* weightValues = weights->values;

    for(uint16_t index:vector->indices){

        // we can only do the chunks of 8 with avx instructions
        // the rest must be done manually
        int size = PARALLEL_SIZE_32_BIT(weights->M);
        // we assume that the output size of the very first layer is always a multiple of 8!
        for(int n = 0; n < size; n+=8){
            // get the gradients into the register aswell as the output which we want to write to
            __m256 wvalues = _mm256_load_ps(&(weightValues[index * weights->M + n]));
            __m256 ovalues = _mm256_load_ps(&(outputValues[                     n]));
            // add the element-wise multiplication of the weights. For this, add the weights for the activated
            // input neuron (output = 1) to the output
            _mm256_store_ps(&outputValues[n],_mm256_add_ps(ovalues, wvalues));
        }
        for(int n = size; n < weights->M; n++){
            outputValues[n] += weightValues[index * weights->M + n];
        }
    }

}

void matmul_backprop(
        Input *vector,
        Data *weights_grad,
        Data *target_grad) {
    assert(target_grad->M == weights_grad->M);
    assert(weights_grad->N % 8 == 0);

    // extract the weight gradient values which we want to compute.
    // there is no input data object as the Sample contains all relevant information.
    // Note that the sample only contains indices for the places where there is a "1"
    // -> there is only 0 or 1 as an input for a neuron
    // The transformation is similar to the backpropagation of the affine transformation
    // which computes gradients for a weights connecting node i with node o by doing:
    // grad(w_io) += output(i) * grad(o)
    float* weightsGrad = weights_grad->values;
    float* outputGrad  =  target_grad->values;

    int size = PARALLEL_SIZE_32_BIT(weights_grad->M);
    // going through each index, applying the rules described above
    // Note that this assumes, as well as the forward step, that the output size is a multiple of 8
    // Otherwise a SIGSEGV will occur as we try to load 256 bit into a register to which we dont have access.
    for(uint16_t &index:vector->indices){
        // we can only do the chunks of 8 with avx instructions
        // the rest must be done manually
        for(int n = 0; n < size; n+=8){
            // get the weight gradient which we want to increment as well as the output gradient
            __m256 wgrad = _mm256_load_ps(&(weightsGrad[index * weights_grad->M + n]));
            __m256 ograd = _mm256_load_ps(&( outputGrad[                          n]));

            _mm256_store_ps(&(weightsGrad[index * weights_grad->M + n]), _mm256_add_ps(wgrad, ograd));
        }
        for(int n = size; n < weights_grad->M; n++){
            weightsGrad[index * weights_grad->M + n] += outputGrad[n];
        }
    }

}