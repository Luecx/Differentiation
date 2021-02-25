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
#include "config.h"

#define ALIGNMENT 64
#define PARALLEL_SIZE_32_BIT(x) x - x %  8

typedef uint16_t Dimension;

class Data{

private:
    bool cleanUp = true;
public:

    float* values = nullptr;
    const int M,N;

    Data(float* values,const int m, const int n=1);
    explicit Data(const int m, const int n=1);
    Data(const Data& other);
    Data(Data&& other) noexcept;
    Data& operator=(const Data &other);
    Data& operator=(Data &&other) noexcept;
    virtual ~Data();

    float  get(int height)                   const;
    float& get(int height);
    float  get(int height, int width)        const;
    float& get(int height, int width);
    float  operator()(int height)            const;
    float& operator()(int height);
    float  operator()(int height, int width) const;
    float& operator()(int height, int width);

    [[nodiscard]] int getM() const;
    [[nodiscard]] int getN() const;
    [[nodiscard]] int size() const;

    void   clear() const;
    void   randomise(float lower=0, float upper=1) const;
    void   add(Data *other);
    void   add(Data *other, float scalar);
    void   sub(Data *other);
    Data  *newInstance() const;

    friend std::ostream& operator<<(std::ostream& os, const Data& data);

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



void matmul(
        const Data*  weights,
        const Input* vector,
        Data*  target,
        int inputOffset);


void matmul_backprop(
        Input    * vector,
        Data     * weights_grad,
        Data     * target_grad,
        int inputOffset);


#endif //DIFFERENTIATION_DATA_H
