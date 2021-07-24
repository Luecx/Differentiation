
#include "Data.h"
#include "DenseLayer.h"
#include "Function.h"
#include "Input.h"
#include "Reader.h"
#include "mappings.h"
#include "network.h"
#include "optimiser.h"

#include <chrono>
#include <algorithm>
#include <random>

constexpr int BATCH_SIZE  = 1024 * 16;

constexpr int IN_SIZE     = 12 * 64;
constexpr int HIDDEN_SIZE = 512;
constexpr int OUTPUT_SIZE = 1;

using namespace dense_relative;

void computeScalars(Network& network, std::vector<Position>& positions){
    Input            inp {};
    Data             output {1};
    float maxHiddenActivation = 0;
    float maxOutputWeight     = std::max(network.getLayer(1)->getWeights()->max(),-network.getLayer(1)->getWeights()->min());
    int idx = 0;
    for(Position& p:positions){
        assign_input(p, inp, output);
        network.evaluate(inp);
        maxHiddenActivation = std::max(maxHiddenActivation, std::max(network.getThreadData(0)->output[0]->max(),-network.getThreadData(0)->output[0]->min()));
        idx += 1;
        if(idx%100000 == 0){
            std::cout << idx << "   " << (1ul << 15) / maxHiddenActivation << "   " << (1ul << 15) / maxOutputWeight << std::endl;
        }
    }
}

void validateFens(Network& network){
    Input            inp {};
    Data             output {1};

    std::string strings[]{
        "8/p4k2/1p3n1p/2pp3P/3P4/1NP2Pr1/PPN1qP2/R5K1 w - - 0 33;0.1",
        "4rrk1/2p1b1p1/p1p3q1/4p3/2P2n1p/1P1NR2P/PB3PP1/3R1QK1 b - - 2 24;0.1",
        "r1bq1rk1/pp2b1pp/n1pp1n2/3P1p2/2P1p3/2N1P2N/PP2BPPP/R1BQ1RK1 b - - 2 10;0.1",
        "r3k2r/ppp1pp1p/2nqb1pn/3p4/4P3/2PP4/PP1NBPPP/R2QK1NR w KQkq - 1 5;0.1",
        "8/1p2pk1p/p1p1r1p1/3n4/8/5R2/PP3PPP/4R1K1 b - - 3 27;0.1",
        "8/8/1p1k2p1/p1prp2p/P2n3P/6P1/1P1R1PK1/4R3 b - - 5 49;0.1",
        "rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq c3 0 2;0.1",
        "8/8/1p4p1/p1p2k1p/P2npP1P/4K1P1/1P6/3R4 w - - 6 54;0.1"
    };

    for(std::string& s:strings){
        Position p {};
        p.set(s);
        assign_input(p, inp, output);

        Data* h = network.evaluate(inp);
        std::cout << s << std::endl;
        std::cout << *h << std::endl;
    }
}

int           main() {

    initLookUpTable();

    std::vector<Position> positions {};

    //    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\eth\stage9.training.fens.100M)", &positions, AGE, 100*1000*1000);
    //    write_positions_bin(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\eth\stage9.training.fens.100M.bin)", &positions);
    //    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\fens_new_mixed.epd)", &positions, AGE);

//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi_4.79-filtered-mix.epd)", &positions, AGE);
//    read_positions_bin(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi_4.79-filtered-mix.bin)", &positions);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\seer\data_fixed_shuff.txt)", &positions, CP_SEER);
//    write_positions_bin(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\seer\data_fixed_shuff.bin)", &positions);
//    read_positions_bin(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\seer\data_fixed_shuff.bin)", &positions);


    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/filtered/00.1.epd)", &positions, AGE, 1000000);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/filtered/00.2.epd)", &positions, AGE);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/filtered/00.3.epd)", &positions, AGE);
//
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/filtered/01.1.epd)", &positions, AGE);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/filtered/01.2.epd)", &positions, AGE);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/filtered/01.3.epd)", &positions, AGE);
//
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/filtered/02.1.epd)", &positions, AGE);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/filtered/02.2.epd)", &positions, AGE);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/filtered/02.3.epd)", &positions, AGE);
//
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/filtered/03.1.epd)", &positions, AGE);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/filtered/03.2.epd)", &positions, AGE);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/filtered/03.3.epd)", &positions, AGE);
//
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/filtered/04.1.epd)", &positions, AGE);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/filtered/04.2.epd)", &positions, AGE);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/filtered/04.3.epd)", &positions, AGE);
//
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/filtered/05.1.epd)", &positions, AGE);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/filtered/05.2.epd)", &positions, AGE);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/filtered/05.3.epd)", &positions, AGE);
//
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/filtered/06.1.epd)", &positions, AGE);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/filtered/06.2.epd)", &positions, AGE);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/filtered/06.3.epd)", &positions, AGE);
//
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/filtered/07.1.epd)", &positions, AGE);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/filtered/07.2.epd)", &positions, AGE);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/filtered/07.3.epd)", &positions, AGE);

//    read_positions_bin(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/shuffled.bin)", &positions);


//    write_positions_bin(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi5.13/shuffled.bin)", &positions);

//    // creating buffers where inputs and targets will be stored for a batch
//    std::vector<Input> inputs {};
//    inputs.resize(BATCH_SIZE);
//
//    std::vector<Data> targets {};
//    for (int i = 0; i < BATCH_SIZE; i++) {
//        targets.emplace_back(Data {OUTPUT_SIZE});
//    }


    auto                         l1 = new DenseLayer<IN_SIZE, HIDDEN_SIZE, ReLU> {};
    auto                         l2 = new DenseLayer<HIDDEN_SIZE, OUTPUT_SIZE, Linear> {};
    std::vector<LayerInterface*> layers {};
    layers.push_back(l1);
    layers.push_back(l2);

    Network network {layers};
    MSEmix     lossFunction {};
//    lossFunction.wdlWeight = 0.5;
    network.setLoss(&lossFunction);
    network.setOptimiser(new Adam());
    network.loadWeights("../resources/networks/koi5.13_relative/23.net");

    validateFens(network);
    computeScalars(network, positions);
//
//    for (int i = 0; i < 1000; i++) {
//
//        int   batch_count = ceil(positions.size() / (float) BATCH_SIZE);
//
//        float lossSum     = 0;
//
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
//
//            printf("\repoch# %-10d batch# %-5d/%-10d loss=%-16.12f speed=%-7d eps", i, batch, batch_count, loss, (int) ((batch * BATCH_SIZE + batchsize) / diff.count()));
//            std::cout << std::flush;
//        }
//        std::cout << std::endl;
//        std::cout << "train loss=" << lossSum / positions.size() << std::endl;
//
//        network.saveWeights("../resources/networks/koi5.13_relative/" + std::to_string(i) + ".net");
//        network.newEpoch();
//    }

    return 0;
}
