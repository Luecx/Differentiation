#include "Data.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include "Function.h"
#include "DenseLayer.h"
#include <new>
#include <omp.h>

int main() {



    Input in{};
    for(int i = 0; i < 512; i++){
        if((i * 12391823ULL) % 10 == 1){
            in.indices.push_back(i);
        }
    }

    Data<8> target{};
    target.randomise(0.7,1);
    auto loss = new MSE<8>{};


    auto l1 = new (std::align_val_t(32)) DenseLayer<512,32,ReLU>{};
    auto l2 = new (std::align_val_t(32)) DenseLayer< 32,32,ReLU>{};
    auto l3 = new (std::align_val_t(32)) DenseLayer< 32, 8,ReLU>{};
    std::vector<LayerInterface*> layers{};
    layers.push_back(l1);
    layers.push_back(l2);
    layers.push_back(l3);


    ThreadData *tds[N_THREADS];
    for(int i = 0; i < N_THREADS; i++){
        tds[i] = new ThreadData{layers};
    }

//


    double first_thread_time = 0;
    for(int t = 1; t < 14; t++){
        for (int j = 0; j < 1; j++) {
            auto start = std::chrono::high_resolution_clock::now();

            float lossSum = 0;
#pragma omp parallel for schedule(auto) num_threads(t) reduction(+:lossSum)
            for (int i = 0; i < 1024*1024*8; i++) {

                const int threadID = omp_get_thread_num();

                l1->apply(&in, tds[threadID]);
                l2->apply(tds[threadID]);
                l3->apply(tds[threadID]);

                lossSum += loss->backprop(dynamic_cast<Data<8> *>(tds[threadID]->output[2]), &target,
                                          dynamic_cast<Data<8> *>(tds[threadID]->output_gradient[2]));
                l3->backprop(tds[threadID]);
                l2->backprop(tds[threadID]);
                l1->backprop(&in, tds[threadID]);


            }

            auto finish = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsed = finish - start;

            double time = elapsed.count();
            if(t == 1){
                first_thread_time = time;
            }

            std::cout << std::left << std::setw(20) << elapsed.count() << "scaling:" << round(first_thread_time / (time * t)*100) << " %" << std::endl;
    }




    }



    return 0;

}
