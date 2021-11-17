
#include "verify/checkGradients.h"
#include "activations/Linear.h"
#include "activations/ReLU.h"
#include "activations/Sigmoid.h"
#include "layers/DenseLayer.h"
#include "layers/DuplicateDenseLayer.h"
#include "loss/MSE.h"
#include "mappings.h"
#include "misc/Reader.h"
#include "network/network.h"
#include "optimiser/Adam.h"
#include "optimiser/optimiser.h"
#include "structures/Data.h"
#include "structures/Input.h"

#include <algorithm>
#include <chrono>
#include <random>

constexpr int BATCH_SIZE   = 8192*2;

constexpr int IN_SIZE      = 12 * 64;
constexpr int HIDDEN1_SIZE = 512;
constexpr int HIDDEN2_SIZE = 32;
constexpr int OUTPUT_SIZE  = 1;

using namespace dense_relative;

#define INPUT_WEIGHT_MULTIPLIER  (256)
#define HIDDEN_WEIGHT_MULTIPLIER (128)

void computeScalars(Network& network, std::vector<Position>& positions) {
    Input inp {};
    Data  output {1};
    float maxHiddenActivation = 0;
    Data maxHiddenActivations = network.getThreadData(0)->output[0]->newInstance();
    float maxOutputWeight     = std::max(network.getLayer(1)->getWeights()->max(), -network.getLayer(1)->getWeights()->min());
    float maxInputWeight      = std::max(network.getLayer(0)->getWeights()->max(), -network.getLayer(0)->getWeights()->min());
    int   idx                 = 0;
    Data  died                = network.getLayer(0)->getBias()->newInstance();
    for (Position& p : positions) {
        assign_input(p, inp, output);
        network.evaluate(inp);
        maxHiddenActivation = std::max(maxHiddenActivation, std::max(network.getThreadData(0)->output[0]->max(), -network.getThreadData(0)->output[0]->min()));

        for(int i = 0; i < network.getThreadData(0)->output[0]->size(); i++){
            maxHiddenActivations.get(i) = std::max(maxHiddenActivations.get(i), std::abs(network.getThreadData(0)->output[0]->get(i)));
        }

        idx += 1;
        if (idx % 100000 == 0) {
            std::cout << idx << "   " << (1ul << 15) / maxHiddenActivation << "   " << (1ul << 15) / maxOutputWeight << std::endl;
        }

//        for (int i = 0; i < died.size(); i++) {
//            died(i) = std::max(died(i), network.getThreadData(0)->output[0]->get(i));
//        }
    }
    std::cout << "max input weight      : " << maxInputWeight << std::endl;
    std::cout << "max hidden activation : " << maxHiddenActivation << std::endl;
    std::cout << "max hidden activations: " << maxHiddenActivations << std::endl;
}

void validateFens(Network& network) {
    Input       inp {};
    Data        output {1};

    std::string strings[] {"2k5/1r6/8/8/7R/8/8/2K5 w - - 0 1;0.1",
                           "8/p4k2/1p3n1p/2pp3P/3P4/1NP2Pr1/PPN1qP2/R5K1 w - - 0 33;0.1",
                           "4rrk1/2p1b1p1/p1p3q1/4p3/2P2n1p/1P1NR2P/PB3PP1/3R1QK1 b - - 2 24;0.1",
                           "r1bq1rk1/pp2b1pp/n1pp1n2/3P1p2/2P1p3/2N1P2N/PP2BPPP/R1BQ1RK1 b - - 2 10;0.1",
                           "r3k2r/ppp1pp1p/2nqb1pn/3p4/4P3/2PP4/PP1NBPPP/R2QK1NR w KQkq - 1 5;0.1",
                           "8/1p2pk1p/p1p1r1p1/3n4/8/5R2/PP3PPP/4R1K1 b - - 3 27;0.1",
                           "8/8/1p1k2p1/p1prp2p/P2n3P/6P1/1P1R1PK1/4R3 b - - 5 49;0.1",
                           "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2;0.1",
                           "8/8/1p4p1/p1p2k1p/P2npP1P/4K1P1/1P6/3R4 w - - 6 54;0.1"};

    for (std::string& s : strings) {
        Position p {};
        p.set(s);
        assign_input(p, inp, output);

        Data* h = network.evaluate(inp);

        std::cout << s << std::endl;
//        std::cout << network.getThreadData(0)->output[0][0] << std::endl;
//        std::cout << network.getLayer(1)->getWeights()[0] << std::endl;
        double check_sum = 0;
//        for(int i = 0; i < 512; i++){
//            check_sum += network.getThreadData(0)->output[0]->get(i) *  network.getLayer(1)->getWeights()->get(i);
//            std::cout << i << "    "
//                      << network.getLayer(1)->getWeights()->get(i) * HIDDEN_WEIGHT_MULTIPLIER << "    "
//                      << network.getThreadData(0)->output[0]->get(i) * INPUT_WEIGHT_MULTIPLIER << "    "
//                      << check_sum  << std::endl;
//        }
        std::cout << *h << std::endl;
//        exit(-1);
    }
}

struct QuantizedWeights{
    int16_t inputWeights[IN_SIZE][HIDDEN1_SIZE] {};
    int16_t hiddenWeights[OUTPUT_SIZE][HIDDEN1_SIZE*2] {};
    int16_t inputBias[HIDDEN1_SIZE] {};
    int32_t hiddenBias[OUTPUT_SIZE] {};
};

void quantitizeNetwork(Network& network, const std::string& output) {
    QuantizedWeights* quantized_weights = new QuantizedWeights();
//    int16_t inputWeights[IN_SIZE][HIDDEN1_SIZE] {};
//    int16_t hiddenWeights[OUTPUT_SIZE][HIDDEN1_SIZE*2] {};
//    int16_t inputBias[HIDDEN1_SIZE] {};
//    int32_t hiddenBias[OUTPUT_SIZE] {};

    // read weights
    int     memoryIndex = 0;
    for (auto & inputWeight : quantized_weights->inputWeights) {
        for (int o = 0; o < HIDDEN1_SIZE; o++) {
            float value        = network.getLayer(0)->getWeights()->get(memoryIndex++);
            inputWeight[o] = round(value * INPUT_WEIGHT_MULTIPLIER);
        }
    }
    // read bias
    memoryIndex = 0;
    for (short & inputBia : quantized_weights->inputBias) {
        float value  = network.getLayer(0)->getBias()->get(memoryIndex++);
        inputBia = round(value * INPUT_WEIGHT_MULTIPLIER);
    }

    // read weights
    memoryIndex = 0;
    for (auto & hiddenWeight : quantized_weights->hiddenWeights) {
        for (int i = 0; i < HIDDEN1_SIZE * 2; i++) {
            float value         = network.getLayer(1)->getWeights()->get(memoryIndex++);
            hiddenWeight[i] = round(value * HIDDEN_WEIGHT_MULTIPLIER);
        }
    }

    // read bias
    memoryIndex = 0;
    for (int & hiddenBia : quantized_weights->hiddenBias) {
        float value   = network.getLayer(1)->getBias()->get(memoryIndex++);
        hiddenBia = round(value * HIDDEN_WEIGHT_MULTIPLIER * INPUT_WEIGHT_MULTIPLIER);
    }

    // write file
    FILE* f = fopen(output.c_str(), "wb");
    fwrite(quantized_weights->inputWeights , sizeof(int16_t), IN_SIZE * HIDDEN1_SIZE, f);
    fwrite(quantized_weights->inputBias    , sizeof(int16_t), HIDDEN1_SIZE, f);
    fwrite(quantized_weights->hiddenWeights, sizeof(int16_t), 2 * HIDDEN1_SIZE * OUTPUT_SIZE, f);
    fwrite(quantized_weights->hiddenBias   , sizeof(int32_t), OUTPUT_SIZE, f);
    fclose(f);

    delete quantized_weights;
}

void analyse_data(std::vector<Position>& positions){
    Data piece_occ[12]{
        Data{8,8},Data{8,8},Data{8,8},Data{8,8},Data{8,8},Data{8,8},
        Data{8,8},Data{8,8},Data{8,8},Data{8,8},Data{8,8},Data{8,8}};
    Data piece_counts{32};
    Data score_counts{10000};
    Data wdl_counts{3};
    for(int i = 0; i < positions.size(); i++){
        PositionIterator iterator{positions[i]};

        // write wdl
        int16_t     tar       = iterator.score;
        if (tar > 10000) {
            wdl_counts(0) += 1;
            tar -= 20000;
        }else if (tar < -10000) {
            wdl_counts(2) += 1;
            tar += 20000;
        }else{
            wdl_counts(1) += 1;
        }
        // write cp
        score_counts(std::clamp(tar + 5000,0,9999)) += 1;

        // go over each piece
        int count = 0;
        while (iterator.hasNext()) {
            iterator.next();
            piece_occ[iterator.piece](mirror(iterator.sq)) += 1;
            count ++;
//            if(iterator.piece == WHITE_KING)
//                wKingSq = it.sq;
//            if(iterator.piece == BLACK_KING)
//                bKingSq = it.sq;
        }
        piece_counts(count) += 1;
    }

    for(int i = 0; i < 12; i++){
        std::cout << piece_occ[i] << std::endl;
    }
    std::cout << piece_counts << std::endl;
    std::cout << wdl_counts << std::endl;
    std::cout << score_counts << std::endl;
}

int main() {

//    int* myArray2 = new int[12]{};
//
//
//    int c = 6;
//
//#pragma omp parallel for reduction(+:myArray2[:c])
//    for (int i=0; i<50; ++i)
//    {
//        double a = 2.0; // Or something non-trivial justifying the parallelism...
//        for (int n = 0; n<6; ++n)
//        {
//            myArray2[n] += a;
//        }
//    }
//    // Print the array elements to see them summed
//    for (int n = 0; n<12; ++n)
//    {
//        std::cout << myArray2[n] << " " << std::endl;
//    }

    initLookUpTable();

    const std::string     path         = R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Differentiation\resources\networks\koi6.24_relative_768-512-2\)";
    const std::string     data_path    = path + "data/";
    const std::string     network_path = path + "networks/";

    // ----------------------------------------------- LOADING DATA ------------------------------------------------------------


//    logging::open(path + "log.txt");

    std::vector<Position> positions {};
//    for(int i = 0; i < 96; i++){
//        read_positions_bin("_" + std::to_string(i), &positions);
//    }
    read_positions_bin(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\netdev\v6.24_generation\kim1.bin)", &positions, 10);
//    analyse_data(positions);
//    exit(-1);
//    read_positions_bin(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\netdev\v6.24_generation\kim2.bin)", &positions);
//    read_positions_bin(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\netdev\v6.24_generation\kim3.bin)", &positions);

    // ----------------------------------------------- DATA SHUFFLING ----------------------------------------------------------

    auto rng = std::default_random_engine {};
    std::shuffle(std::begin(positions), std::end(positions), rng);

    // ----------------------------------------------- BATCH PREPARATION ------------------------------------------------------------

    // creating buffers where inputs and targets will be stored for a batch
    std::vector<Input> inputs {};
    inputs.resize(BATCH_SIZE);

    std::vector<Data> targets {};
    for (int i = 0; i < BATCH_SIZE; i++) {
        targets.emplace_back(Data {OUTPUT_SIZE});
    }

    // ----------------------------------------------- NETWORK STRUCTURE ------------------------------------------------------------

    auto                         l1 = new DuplicateDenseLayer<  IN_SIZE     , HIDDEN1_SIZE, ReLU> {};
    auto                         l2 = new DenseLayer         <2*HIDDEN1_SIZE, 1, Sigmoid> {};
//    auto                         l3 = new DenseLayer         <  HIDDEN2_SIZE, OUTPUT_SIZE , Sigmoid> {};
    std::vector<LayerInterface*> layers {};
    layers.push_back(l1);
    layers.push_back(l2);
//    layers.push_back(l3);

    Network network {layers};
    MSE     lossFunction {};
    Adam    adam {};
    network.setLoss(&lossFunction);
    network.setOptimiser(&adam);

    network.logOverview();

//    network.trackSparseInputsOverBatch(true);


    // ----------------------------------------------- VALIDATION ------------------------------------------------------------
//    network.loadWeights(R"(C:\Users\Luecx\CLionProjects\Differentiation\resources\networks\koi5.13_relative_768-512-1\37_gd.net)");
//    network.loadWeights(network_path + "42.nn");
//    quantitizeNetwork(network, network_path + "42.net");
//    validateFens(network);
//    computeScalars(network, positions);
//    exit(-1);
//    // ----------------------------------------------- TRAINING ------------------------------------------------------------
    for (int i = 1; i < 100; i++) {
//        network.loadWeights(network_path + std::to_string(i) + ".nn");
        int   batch_count = ceil(positions.size() / (float) BATCH_SIZE);
        float lossSum     = 0;
        auto  start       = std::chrono::system_clock::now();

        for (int batch = 0; batch < batch_count; batch++) {
            // fill the inputs and outputs
            int   batchsize = assign_inputs_batch(positions, batch * BATCH_SIZE, inputs, targets);
            float loss      = network.batch(inputs, targets, batchsize, true);

            lossSum += loss * batchsize;
            auto                          end  = std::chrono::system_clock::now();
            std::chrono::duration<double> diff = end - start;
            printf("\repoch# %-10d batch# %-5d/%-10d loss=%-16.12f speed=%-7d eps", i, batch, batch_count, loss, (int) ((batch * BATCH_SIZE + batchsize) / diff.count()));
            std::cout << std::flush;
        }

        std::stringstream ss{};
        ss
                 << "epoch: " << std::left << std::setw(10) << i
                 << "loss: "  << std::left << std::setw(10) << lossSum / positions.size();
        logging::write(ss.str());


        std::cout << std::endl;
        std::cout << " train loss=" << lossSum / positions.size() << std::endl;

//        network.saveWeights(network_path + std::to_string(i) + ".nn");
        network.newEpoch();
    }



//    network.loadWeights(R"(..\resources\networks\koi6.24_half_kingside_512_mirror\networks\44.nn)");
//    for(int n = 0; n < HIDDEN1_SIZE; n++){
//        for(int i = 0; i < network.getLayer(0)->getWeights()->getN(); i++){
//            network.getLayer(0)->getWeights()->get(n, i) /= 12;
//        }
//        network.getLayer(0)->getBias()->get(n) /= 12;
//        network.getLayer(1)->getWeights()->get(n) *= 12;
//        network.getLayer(1)->getWeights()->get(n+HIDDEN1_SIZE) *= 12;
//    }
//    quantitizeNetwork(network, R"(..\resources\networks\koi6.24_half_kingside_512_mirror\networks\44.net)");
//    validateFens(network);
//    computeScalars(network, positions);



    return 0;
}
