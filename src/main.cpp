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
#include <algorithm>
#include "network.h"
#include "DuplicateDenseLayer.h"
#include <random>

#include "mappings.h"


int main() {

    int BATCH_SIZE = 4096;

    initLookUpTable();

//    std::vector<ataxx::Sample> positions{};
//    std::vector<ataxx::Sample> validation{};
//    ataxx::load_positions("../resources/part-1.txt", positions);
//    ataxx::load_positions("../resources/part-2.txt", positions);
//    ataxx::load_positions("../resources/part-3.txt", positions);
//    ataxx::load_positions("../resources/part-4.txt", positions);
//    ataxx::load_positions("../resources/part-5.txt", positions);
//    ataxx::load_positions("../resources/validation.txt", validation);
    // loading positions
    std::vector<Position> positions{};
//    positions.reserve(300 * 1000 * 1000);
//    read_positions_bin("position.bin",&positions);
    read_positions_txt("F:\\OneDrive\\ProgrammSpeicher\\CLionProjects\\Koivisto\\resources\\make_fens_from_lichess_pgn\\output\\0.txt", &positions,3000000);
    std::cout << "beginning shuffling" << std::endl;
    std::shuffle(positions.begin(), positions.end(), std::random_device());
    std::cout << "finished shuffling" << std::endl;

    // creating buffers where inputs and targets will be stored for a batch
    std::vector<Input> inputs{};
    inputs.resize(BATCH_SIZE);

    std::vector<Data> targets{};
    for (int i = 0; i < BATCH_SIZE; i++) {
        targets.emplace_back(Data{1});
    }

    // setting up the network
    constexpr int IN_SIZE       = 12*64;
    constexpr int HIDDEN1_SIZE  = 512;
    constexpr int OUTPUT_SIZE   = 1;

    auto l1 = new DenseLayer<IN_SIZE, HIDDEN1_SIZE, ReLU>{};
    auto l2 = new DenseLayer<HIDDEN1_SIZE, OUTPUT_SIZE, Sigmoid>{};
    std::vector<LayerInterface *> layers{};
    layers.push_back(l1);
    layers.push_back(l2);

    Network network{layers};
    network.setLoss(new MSE());
    network.setOptimiser(new Adam());

    network.loadWeights("halogen.net");

//    ataxx::Sample test_pos{};
//    Input         test_pos_in{};
//    Data          test_pos_tar{1};
//    test_pos.set("0 x5o/7/7/7/7/7/o5x x 0 1");
//    ataxx::assign_input(test_pos, test_pos_in, test_pos_tar);
//    std::cout << *network.evaluate(test_pos_in) << std::endl;

    for (int i = 0; i < 300; i++) {

        int batch_count = ceil(positions.size() / (float) BATCH_SIZE);

        float lossSum = 0;

        auto start = std::chrono::system_clock::now();

        for (int batch = 0; batch < batch_count; batch++) {
            // fill the inputs and outputs
            int batchsize = dense::assign_inputs_batch(positions, batch * BATCH_SIZE, inputs, targets);
            float loss = network.batch(inputs, targets, batchsize);
            lossSum += loss * batchsize;
            auto end = std::chrono::system_clock::now();
            std::chrono::duration<double> diff = end - start;
            printf("\repoch# %-10d batch# %-5d/%-10d loss=%-16.12f speed=%-7d eps",
                   i, batch, batch_count, loss,
                   (int) ((batch * BATCH_SIZE + batchsize) / diff.count()));
            std::cout << std::flush;
        }
        std::cout << std::endl;
        std::cout << "train loss=" << lossSum / positions.size() << std::endl;





//        batch_count = ceil(validation.size() / (float) BATCH_SIZE);
//
//        lossSum = 0;
//        start = std::chrono::system_clock::now();
//        for (int batch = 0; batch < batch_count; batch++) {
//            // fill the inputs and outputs
//            int batchsize = ataxx::assign_inputs_batch(validation, batch * BATCH_SIZE, inputs, targets);
//            float loss = network.batch(inputs, targets, batchsize, false);
//            lossSum += loss * batchsize;
//            auto end = std::chrono::system_clock::now();
////            printf("\repoch# %-10d batch# %-5d/%-10d loss=%-16.12f speed=%-7d eps",
////                   i, batch, batch_count, loss,
////                   (int) ((batch * BATCH_SIZE + batchsize) / lossSum.count()));
////            std::cout << std::flush;
//        }
//        std::cout << std::endl;
//        std::cout << "validation loss=" << lossSum / validation.size() << std::endl;





        network.newEpoch();
//        network.saveWeights("halogen.net");
    }


    return 0;

}
