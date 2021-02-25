#include "Data.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include "Function.h"
#include "DenseLayer.h"
#include "optimiser.h"
#include "merge.h"
#include "Reader.h"
#include <new>
#include <omp.h>
#include <bitset>
#include "network.h"
#include "DuplicateDenseLayer.h"

int main() {




    constexpr int IN_SIZE = 64*6*64;
    constexpr int HIDDEN1_SIZE = 256;
    constexpr int HIDDEN2_SIZE = 32;
    constexpr int HIDDEN3_SIZE = 32;
    constexpr int OUTPUT_SIZE = 1;

    auto l1 = new DuplicateDenseLayer<IN_SIZE, HIDDEN1_SIZE, ReLU>{};
    auto l2 = new DenseLayer<HIDDEN1_SIZE * 2, HIDDEN2_SIZE, ReLU>{};
    auto l3 = new DenseLayer<HIDDEN2_SIZE, HIDDEN3_SIZE, ReLU>{};
    auto l4 = new DenseLayer<HIDDEN3_SIZE, OUTPUT_SIZE, Linear>{};
//    l4->bias.randomise(10,10);
    std::vector<LayerInterface *> layers{};
    layers.push_back(l1);
    layers.push_back(l2);
    layers.push_back(l3);
    layers.push_back(l4);

    Network network{layers};
    network.setLoss(new MSE());
    network.setOptimiser(new Adam());





//    ThreadData* td = new ThreadData(0,layers);
//    l4->apply(td);
//
//    std::cout << *td->output[3] << std::endl;



    Input in1{};
    for (int i = 0; i < IN_SIZE*2; i++) {
        if ((i * 12391823ULL) % 1000 == 1) {
            in1.indices.push_back(i);
        }
    }

    Data target{OUTPUT_SIZE};
    target.randomise(0.7, 1);

    std::vector<Input> inputs{};

    std::vector<Data> outputs{};

    for(int i = 0; i < 4096; i++){
        inputs.emplace_back(in1);
        outputs.push_back(target);
    }


    for(int i = 0; i < 1000; i++){
        auto                         start = std::chrono::system_clock::now();
        float                         loss = network.batch(inputs, outputs);
        auto                          end  = std::chrono::system_clock::now();
        std::chrono::duration<double> diff = end - start;
        printf("\repoch# %-10d batch# %-10d loss=%-16.12f speed=%-7d eps", i, 0, loss, (int)(inputs.size() / diff.count()));
        std::cout << std::endl;
        network.newEpoch();
    }


    return 0;

}
