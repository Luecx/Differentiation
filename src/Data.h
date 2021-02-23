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
#include <cassert>
#include "Input.h"
#include "util.h"

#define ALIGNMENT 64
#define N_THREADS 32
#define PARALLEL_SIZE_32_BIT(x) x - x %  8

typedef uint16_t Dimension;

class Data{

private:
    bool cleanUp = true;
public:

    float* values = nullptr;
    const int M,N;

    Data(float* values,const int m, const int n=1) : M(m), N(n) {
        cleanUp = false;
        this->values = values;
    }
    Data(const int m, const int n=1) : M(m), N(n) {
        this->values = (float*) _mm_malloc(M*N * sizeof(float), ALIGNMENT);
//        this->values = new (std::align_val_t(ALIGNMENT)) float[M*N] {};
    }
    Data(const Data& other): M(other.M), N(other.N){
        this->values = (float*) _mm_malloc(M*N * sizeof(float), ALIGNMENT);
//        this->values = new (std::align_val_t(ALIGNMENT)) float[M*N] {};
        std::memcpy(values, other.values, sizeof(float) * M * N);
    }
    Data(Data&& other): M(other.M), N(other.N){
        this->values = other.values;
        other.values = nullptr;
    }
    Data& operator=(const Data &other){
        assert(other.M == M && other.N == N);
        std::memcpy(values, other.values, sizeof(float) * M * N);
        return *this;
    }
    Data& operator=(Data &&other) noexcept {
        assert(other.M == M && other.N == N);
        this->values = other.values;
        other.values = nullptr;
        return *this;
    }
    virtual ~Data() {
        if(this->values != nullptr && cleanUp){
            _mm_free(this->values);
            this->values = nullptr;
        }
    }

    float  get(int height)                   const { return values[height]; }
    float& get(int height)                         { return values[height]; }
    float  get(int height, int width)        const { return values[width + height * N]; }
    float& get(int height, int width)              { return values[width + height * N]; }
    float  operator()(int height)            const { return get(height); }
    float& operator()(int height)                  { return get(height); }
    float  operator()(int height, int width) const { return get(height, width); }
    float& operator()(int height, int width)       { return get(height, width); }

    [[nodiscard]] int getM() const {
        return M;
    }
    [[nodiscard]] int getN() const {
        return N;
    }

    void   clear() const{
        if(values != nullptr)
            std::memset(values, 0, sizeof(float) * M * N);
    }
    void   randomise(float lower=0, float upper=1) const {
        for (int i = 0; i < M*N; i++) {
            this->values[i] = static_cast<float>(rand()) / RAND_MAX * (upper - lower) + lower;
        }
    }
    void   add(Data *other){
        assert(other->M == M && other->N == N);
        const int size = PARALLEL_SIZE_32_BIT(M*N);
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
    void   add(Data *other, float scalar){
        assert(other->M == M && other->N == N);
        const int size = PARALLEL_SIZE_32_BIT(M*N);
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
    void   sub(Data *other){
        assert(other->M == M && other->N == N);
        const int size = PARALLEL_SIZE_32_BIT(M*N);
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
    Data  *newInstance() const {
        return new Data{M,N};
    }

    friend std::ostream& operator<<(std::ostream& os, const Data& data) {

        if(data.N != 1){
            os << std::fixed << std::setprecision(3);
            for (int i = 0; i < data.M; i++) {
                for (int n = 0; n < data.N; n++) {
                    os << std::setw(11) << (double)data(i,n);
                }
                os << "\n";
            }
        }else{
            os << "(transposed) ";
            for (int n = 0; n < data.M; n++) {
                os << std::setw(11) << (double)data(n);
            }
        }
        return os;

    }

};

void matmul(
        const Data* weights,
        const Data* vector,
        Data* target);

void matmul_backprop(
        const Data* weights,
        const Data* vector,
        Data* weights_grad,
        Data* vector_grad,
        const Data* target_grad);


void matmul(
        const Data*  weights,
        const Input* vector,
        Data*  target);


void matmul_backprop(
        Input    * vector,
        Data     * weights_grad,
        Data     * target_grad);



#endif //DIFFERENTIATION_DATA_H
