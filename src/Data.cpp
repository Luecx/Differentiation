#include "Data.h"

#include <algorithm>
#include <random>

Data::Data(float* values, const int m, const int n) : M(m), N(n) {
    cleanUp      = false;
    this->values = values;
}

Data::Data(const int m, const int n) : M(m), N(n) {
    this->values = (float*) _mm_malloc(M * N * sizeof(float), ALIGNMENT);
    this->clear();
    //        this->values = new (std::align_val_t(ALIGNMENT)) float[M*N] {};
}

Data::Data(const Data& other) : M(other.M), N(other.N) {
    this->values = (float*) _mm_malloc(M * N * sizeof(float), ALIGNMENT);
    //        this->values = new (std::align_val_t(ALIGNMENT)) float[M*N] {};
    std::memcpy(values, other.values, sizeof(float) * M * N);
}

Data::Data(Data&& other) noexcept : M(other.M), N(other.N) {
    this->values = other.values;
    other.values = nullptr;
}

Data& Data::operator=(const Data& other) {
    assert(other.M == M && other.N == N);
    std::memcpy(values, other.values, sizeof(float) * M * N);
    return *this;
}

Data& Data::operator=(Data&& other) noexcept {
    assert(other.M == M && other.N == N);
    this->values = other.values;
    other.values = nullptr;
    return *this;
}

Data::~Data() {
    if (this->values != nullptr && cleanUp) {
        _mm_free(this->values);
        this->values = nullptr;
    }
}

float  Data::get(int height) const { return values[height]; }

float& Data::get(int height) { return values[height]; }

float  Data::get(int height, int width) const { return values[width + height * N]; }

float& Data::get(int height, int width) { return values[width + height * N]; }

float Data::operator()(int height) const { return get(height); }

float& Data::operator()(int height) { return get(height); }

float Data::operator()(int height, int width) const { return get(height, width); }

float& Data::operator()(int height, int width) { return get(height, width); }

float        Data::min() const {
    float m = values[0];
    for (int i = 0; i < size(); i++) {
        m = std::min(m, values[i]);
    }
    return m;
}
float Data::max() const {

    float m = values[0];
    for (int i = 0; i < size(); i++) {
        m = std::max(m, values[i]);
    }
    return m;
}

void Data::sort() { std::sort(values, values + size(), std::greater<float>()); }

int  Data::getM() const { return M; }

int  Data::getN() const { return N; }

int  Data::size() const { return M * N; }

void Data::clear() const {
    if (values != nullptr)
        std::memset(values, 0, sizeof(float) * M * N);
}

void Data::randomise(float lower, float upper) const {
    for (int i = 0; i < M * N; i++) {
        this->values[i] = static_cast<float>(rand()) / RAND_MAX * (upper - lower) + lower;
    }
}

void Data::randomiseGaussian(float mean, float deviation) {
    std::default_random_engine      generator;
    std::normal_distribution<float> distribution(0, deviation);
    for (int i = 0; i < M * N; i++) {
        this->values[i] = distribution(generator);
    }
}

void Data::randomiseKieren() {
#define uniform() ((float) (rand() + 1) / ((float) RAND_MAX + 2))
#define random()  (sqrtf(-2.0 * log(uniform())) * cos(2 * M_PI * uniform()))

    for (int j = 0; j < M * N; j++)
        values[j] = random() / 4.0;

#undef uniform
#undef random
}

void Data::scale(float scale) {
    for (int i = 0; i < M * N; i++) {
        this->values[i] *= scale;
    }
}

void Data::add(Data* other) {
    assert(other->M == M && other->N == N);
    const int size = PARALLEL_SIZE_32_BIT(M * N);
    for (int i = 0; i < size; i += 8) {
        // load our values and the target values into the register
        __m256* other_values = (__m256*) (&other->values[i]);
        __m256* our_values   = (__m256*) (&this->values[i]);
        // stores the sum of our and their values inside the other data object.
        *our_values          = _mm256_add_ps(*other_values, *our_values);
    }
    for (int i = size; i < M * N; i++) {
        (*this)(i) += (*other)(i);
    }
}

void Data::add(Data* other, float scalar) {
    assert(other->M == M && other->N == N);
    const int size = PARALLEL_SIZE_32_BIT(M * N);
    __m256    s    = _mm256_set1_ps(scalar);
    for (int i = 0; i < size; i += 8) {
        // load our values and the target values into the register
        __m256* other_values = (__m256*) (&other->values[i]);
        __m256* our_values   = (__m256*) (&this->values[i]);
        // stores the sum of our and their values inside the other data object.
        *our_values          = _mm256_add_ps(_mm256_mul_ps(s, *other_values), *our_values);
    }
    for (int i = size; i < M * N; i++) {
        (*this)(i) += (*other)(i) *scalar;
    }
}

void Data::sub(Data* other) {
    assert(other->M == M && other->N == N);
    const int size = PARALLEL_SIZE_32_BIT(M * N);
    for (int i = 0; i < size; i += 8) {
        // load our values and the target values into the register
        __m256* other_values = (__m256*) (&other->values[i]);
        __m256* our_values   = (__m256*) (&this->values[i]);
        // stores the sum of our and their values inside the other data object.
        *our_values          = _mm256_sub_ps(*our_values, *other_values);
    }
    for (int i = size; i < M * N; i++) {
        (*this)(i) -= (*other)(i);
    }
}

Data   Data::newInstance() const{
    return Data{this->M, this->N};
}

std::ostream& operator<<(std::ostream& os, const Data& data) {

    if (data.N != 1) {
        os << std::fixed << std::setprecision(5);
        for (int i = 0; i < data.M; i++) {
            for (int n = 0; n < data.N; n++) {
                os << std::setw(11) << (double) data(i, n);
            }
            os << "\n";
        }
    } else {
        os << "(transposed) ";
        for (int n = 0; n < data.M; n++) {
            os << std::setw(11) << (double) data(n);
        }
    }
    return os;
}