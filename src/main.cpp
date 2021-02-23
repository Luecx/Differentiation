#include "Data.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include "Function.h"
#include "DenseLayer.h"
#include "optimiser.h"
#include "merge.h"
#include <new>
#include <omp.h>

int main() {


    constexpr int IN_SIZE = 64*6*64;
    constexpr int HIDDEN1_SIZE = 256;
    constexpr int HIDDEN2_SIZE = 32;
    constexpr int HIDDEN3_SIZE = 32;
    constexpr int OUTPUT_SIZE = 1;


    Input in1{};
    for (int i = 0; i < IN_SIZE; i++) {
        if ((i * 12391823ULL) % 1000 == 1) {
            in1.indices.push_back(i);
        }
    }
    Input in2{};
    for (int i = 0; i < IN_SIZE; i++) {
        if ((i * 12391823ULL) % 1000 == 1) {
            in2.indices.push_back(i);
        }
    }

    Data target{OUTPUT_SIZE};
    target.randomise(0.7, 1);
    auto loss = new MSE{};

    auto l1 = new(std::align_val_t(32)) DuplicateDenseLayer<IN_SIZE, HIDDEN1_SIZE, ReLU>{};
    auto l2 = new(std::align_val_t(32)) DenseLayer<HIDDEN1_SIZE * 2, HIDDEN2_SIZE, ReLU>{};
    auto l3 = new(std::align_val_t(32)) DenseLayer<HIDDEN2_SIZE, HIDDEN3_SIZE, ReLU>{};
    auto l4 = new(std::align_val_t(32)) DenseLayer<HIDDEN3_SIZE, OUTPUT_SIZE, ReLU>{};
    std::vector<LayerInterface *> layers{};
    layers.push_back(l1);
    layers.push_back(l2);
    layers.push_back(l3);
    layers.push_back(l4);

    ThreadData *tds[N_THREADS];
    for (int i = 0; i < N_THREADS; i++) {
        tds[i] = new ThreadData{i, layers};
    }
    for (LayerInterface *l:layers) {
        l->assignThreadData(tds);
    }

    Adam adam{layers};
    auto start = std::chrono::high_resolution_clock::now();
    for(int i = 0; i < 100; i++){
        merge_gradients(tds);
//        adam.apply(tds[0]);
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "elapsed time:" << elapsed.count() << std::endl;
//
//    double first_thread_time = 0;
////    for (int t = 1; t < 16; t++) {
//        for (int j = 0; j < 1; j++) {
////            auto start = std::chrono::high_resolution_clock::now();
//
//            float lossSum = 0;
////#pragma omp parallel for schedule(auto) num_threads(1) reduction(+:lossSum)
//            for (int i = 0; i < 100; i++) {
//
//                const int threadID = 0;
//
//                l1->apply(&in1, &in2, tds[threadID]);
//                l2->apply(tds[threadID]);
//                l3->apply(tds[threadID]);
//                l4->apply(tds[threadID]);
//
//                lossSum = loss->backprop((tds[threadID]->output[3]), &target,
//                                          (tds[threadID]->output_gradient[3]));
//                l4->backprop(tds[threadID]);
//                l3->backprop(tds[threadID]);
//                l2->backprop(tds[threadID]);
//                l1->backprop(&in1, &in2, tds[threadID]);
//
//                std::cout << lossSum << std::endl;
//                l4->weights.add(tds[0]->weight_gradient[3], -0.01);
//                tds[0]->weight_gradient[3]->clear();
////                std::cout << *tds[0]->output_gradient[3] << std::endl;
////                std::cout << *tds[0]->output[3] << std::endl;
//            }
//
////            auto finish = std::chrono::high_resolution_clock::now();
////            std::chrono::duration<double> elapsed = finish - start;
////
////            double time = elapsed.count();
////            if (t == 1) {
////                first_thread_time = time;
////            }
////
////            std::cout << std::left << std::setw(20) << elapsed.count() << "scaling:"
////                      << round(first_thread_time / (time * t) * 100) << " %" << std::endl;
//        }
////    }


    return 0;

}
