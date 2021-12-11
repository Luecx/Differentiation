
#include "activations/ReLU.h"
#include "activations/Sigmoid.h"
#include "dataset/batchloader.h"
#include "dataset/reader.h"
#include "layers/DenseLayer.h"
#include "layers/DuplicateDenseLayer.h"
#include "loss/MSE.h"
#include "mappings.h"
#include "network/network.h"
#include "optimiser/Adam.h"
#include "optimiser/optimiser.h"
#include "structures/Data.h"
#include "structures/Input.h"
#include "verify/checkGradients.h"

#include <algorithm>
#include <chrono>
#include <random>

constexpr int BATCH_SIZE   = 1024 * 16;
constexpr int EPOCH_SIZE   = 1024 * 16;

constexpr int IN_SIZE      = 12 * 64;
constexpr int HIDDEN1_SIZE = 512;
constexpr int OUTPUT_SIZE  = 1;

using namespace dense_relative;

#define INPUT_WEIGHT_MULTIPLIER  (64)
#define HIDDEN_WEIGHT_MULTIPLIER (512)

void computeScalars(Network& network, std::vector<Position>& positions) {
    Input inp {};
    Data  output {1};
    float maxHiddenActivation = 0;
    float maxOutputWeight     = std::max(network.getLayer(1)->getWeights()->max(), -network.getLayer(1)->getWeights()->min());
    int   idx                 = 0;
    Data  died                = network.getLayer(0)->getBias()->newInstance();
    for (Position& p : positions) {
        assign_input(p, inp, output);
        network.evaluate(inp);
        maxHiddenActivation = std::max(maxHiddenActivation, std::max(network.getThreadData(0)->output[0]->max(), -network.getThreadData(0)->output[0]->min()));
        idx += 1;
        if (idx % 100000 == 0) {
            std::cout << idx << "   " << (1ul << 14) / maxHiddenActivation << "   " << (1ul << 14) / maxOutputWeight << std::endl;
        }

        for (int i = 0; i < died.size(); i++) {
            died(i) = std::max(died(i), network.getThreadData(0)->output[0]->get(i));
        }
    }
    std::cout << "max hidden activation : " << maxHiddenActivation << std::endl;
    std::cout << "max hidden activations: " << died << std::endl;
}

void validateFens(Network& network) {
    Input       inp {};
    Data        output {1};

    std::string strings[] {"8/p4k2/1p3n1p/2pp3P/3P4/1NP2Pr1/PPN1qP2/R5K1 w - - 0 33;0.1",
                           "4rrk1/2p1b1p1/p1p3q1/4p3/2P2n1p/1P1NR2P/PB3PP1/3R1QK1 b - - 2 24;0.1",
                           "r1bq1rk1/pp2b1pp/n1pp1n2/3P1p2/2P1p3/2N1P2N/PP2BPPP/R1BQ1RK1 b - - 2 10;0.1",
                           "r3k2r/ppp1pp1p/2nqb1pn/3p4/4P3/2PP4/PP1NBPPP/R2QK1NR w KQkq - 1 5;0.1",
                           "8/1p2pk1p/p1p1r1p1/3n4/8/5R2/PP3PPP/4R1K1 b - - 3 27;0.1",
                           "8/8/1p1k2p1/p1prp2p/P2n3P/6P1/1P1R1PK1/4R3 b - - 5 49;0.1",
                           "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2;0.1",
                           "8/8/1p4p1/p1p2k1p/P2npP1P/4K1P1/1P6/3R4 w - - 6 54;0.1"};

    for (std::string& s : strings) {
        Position p {};
//        p.set(s);
        assign_input(p, inp, output);

        Data* h = network.evaluate(inp);
//        std::cout << network.getThreadData(0)->output[0][0];
        std::cout << s << std::endl;
        for(int i = 0; i < 16; i++){
            std::cout << network.getThreadData(0)->output[0]->get(i) * INPUT_WEIGHT_MULTIPLIER << "  ";
        }
        std::cout << std::endl;
        for(int i = 0; i < 16; i++){
            std::cout << network.getThreadData(0)->output[0]->get(i + HIDDEN1_SIZE) * INPUT_WEIGHT_MULTIPLIER<< "  ";
        }
        std::cout << std::endl;

        std::cout << *h << std::endl;
        exit(-1);
    }
}

void quantitizeNetwork(Network& network, const std::string& output) {

    int16_t inputWeights[IN_SIZE][HIDDEN1_SIZE] {};
    int16_t hiddenWeights[OUTPUT_SIZE][HIDDEN1_SIZE] {};
    int16_t inputBias[HIDDEN1_SIZE] {};
    int32_t hiddenBias[OUTPUT_SIZE] {};

    // read weights
    int     memoryIndex = 0;
    for (auto & inputWeight : inputWeights) {
        for (int o = 0; o < HIDDEN1_SIZE; o++) {
            float value        = network.getLayer(0)->getWeights()->get(memoryIndex++);
            inputWeight[o] = round(value * INPUT_WEIGHT_MULTIPLIER);
        }
    }
    // read bias
    memoryIndex = 0;
    for (short & inputBia : inputBias) {
        float value  = network.getLayer(0)->getBias()->get(memoryIndex++);
        inputBia = round(value * INPUT_WEIGHT_MULTIPLIER);
    }

    // read weights
    memoryIndex = 0;
    for (auto & hiddenWeight : hiddenWeights) {
        for (int i = 0; i < HIDDEN1_SIZE; i++) {
            float value         = network.getLayer(1)->getWeights()->get(memoryIndex++);
            hiddenWeight[i] = round(value * HIDDEN_WEIGHT_MULTIPLIER);
        }
    }

    // read bias
    memoryIndex = 0;
    for (int & hiddenBia : hiddenBias) {
        float value   = network.getLayer(1)->getBias()->get(memoryIndex++);
        hiddenBia = round(value * HIDDEN_WEIGHT_MULTIPLIER * INPUT_WEIGHT_MULTIPLIER);
    }

    // write file
    FILE* f = fopen(output.c_str(), "wb");
    fwrite(inputWeights, sizeof(int16_t), IN_SIZE * HIDDEN1_SIZE, f);
    fwrite(inputBias, sizeof(int16_t), HIDDEN1_SIZE, f);
    fwrite(hiddenWeights, sizeof(int16_t), HIDDEN1_SIZE * OUTPUT_SIZE, f);
    fwrite(hiddenBias, sizeof(int32_t), OUTPUT_SIZE, f);
    fclose(f);
}

int main() {

    const std::string     path         = "../runs/king_relative_256/";
    const std::string     data_path    = path + "data/";
    const std::string     network_path = path + "networks/";

    // ----------------------------------------------- LOADING DATA ------------------------------------------------------------


    std::vector<std::string> files {};

    files.push_back(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi7.9\generated_0.txt.bin)");
    BatchLoader batch_loader{files, BATCH_SIZE, 128};
    // ----------------------------------------------- BATCH PREPARATION ------------------------------------------------------------

    // creating buffers where inputs and targets will be stored for a batch
    std::vector<Input> inputs {};
    inputs.resize(BATCH_SIZE);

    std::vector<Data> targets {};
    for (int i = 0; i < BATCH_SIZE; i++) {
        targets.emplace_back(Data {OUTPUT_SIZE});
    }


    // ----------------------------------------------- NETWORK STRUCTURE ------------------------------------------------------------

    auto                         l1 = new DuplicateDenseLayer<IN_SIZE, HIDDEN1_SIZE, ReLU> {64};
    auto                         l2 = new DenseLayer         <HIDDEN1_SIZE * 2, 1, Sigmoid> {};
    std::vector<LayerInterface*> layers {};
    layers.push_back(l1);
    layers.push_back(l2);

    Network network {layers};
    MSE     lossFunction {};
    Adam    adam {};
    adam.alpha = 0.01;
    network.setLoss(&lossFunction);
    network.setOptimiser(&adam);

    network.logOverview();

//    Input input{};
//    Data data{1};
//    data.get(0) = 0.3;
//    check_gradients(network, input, data, 0);
//    check_gradients(network, input, data, 1);
//    check_gradients(network, input, data, 2);

    // ----------------------------------------------- VALIDATION ------------------------------------------------------------
//    network.loadWeights(R"(C:\Users\Luecx\CLionProjects\Differentiation\resources\networks\koi5.13_relative_768-512-1\37_gd.net)");
//    network.loadWeights(network_path + "60.nn");
//    quantitizeNetwork(network, network_path + "42.net");
//    validateFens(network);
//    computeScalars(network, positions);
//    exit(-1);
    // ----------------------------------------------- TRAINING ------------------------------------------------------------
//
    for(int epoch = 0; epoch < 1; epoch ++){

        float lossSum     = 0;
        auto  start       = std::chrono::system_clock::now();

        for(int batch = 0; batch < EPOCH_SIZE; batch++){
            DataSet* position  = batch_loader.next();

            // fill the inputs and outputs
            int      batchsize = assign_inputs_batch(*position, inputs, targets);
            float    loss      = network.batch(inputs, targets, batchsize, true);

            lossSum += loss * batchsize;
            auto                          end  = std::chrono::system_clock::now();
            std::chrono::duration<double> diff = end - start;
            printf("\repoch# %-10d batch# %-5d/%-10d loss=%-16.12f avg. loss=%-16.12f speed=%-7d eps",
                   epoch,
                   batch,
                   EPOCH_SIZE,
                   loss,
                   lossSum / batch / batchsize,
                   (int) ((batch * BATCH_SIZE + batchsize) / diff.count()));
            std::cout << std::flush;
        }
    }

//    for (int i = 1; i < 20000; i++) {
////        network.loadWeights(network_path + std::to_string(i) + ".nn");
//
//        int   batch_count = ceil(positions.size() / (float) BATCH_SIZE);
//        float lossSum     = 0;
//        auto  start       = std::chrono::system_clock::now();
//
//        for (int batch = 0; batch < batch_count; batch++) {
//            // fill the inputs and outputs
//            int   batchsize = assign_inputs_batch(positions, batch * BATCH_SIZE, inputs, targets);
//            float loss      = network.batch(inputs, targets, batchsize, true);
//
//            lossSum += loss * batchsize;
//            auto                          end  = std::chrono::system_clock::now();
//            std::chrono::duration<double> diff = end - start;
//            printf("\repoch# %-10d batch# %-5d/%-10d loss=%-16.12f speed=%-7d eps", i, batch, batch_count, loss, (int) ((batch * BATCH_SIZE + batchsize) / diff.count()));
//            std::cout << std::flush;
//        }
//
//        std::stringstream ss{};
//        ss
//                 << "epoch: " << std::left << std::setw(10) << i
//                 << "loss: "  << std::left << std::setw(10) << lossSum / positions.size();
//        logging::write(ss.str());
//
//
//        std::cout << std::endl;
//        std::cout << " train loss=" << lossSum / positions.size() << std::endl;
//
////        network.saveWeights(network_path + std::to_string(i) + "._lrdrop1.nn");
//        network.newEpoch();
//    }
//    network.loadWeights("test.nn");
//    quantitizeNetwork(network, "test.net");
//    validateFens(network);
//    computeScalars(network, positions);

    return 0;
}
