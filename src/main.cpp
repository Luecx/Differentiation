
#include "Data.h"
#include "DenseLayer.h"
#include "Function.h"
#include "Input.h"
#include "Reader.h"
#include "mappings.h"
#include "network.h"
#include "optimiser.h"

#include <chrono>


constexpr int BATCH_SIZE  = 1024;

constexpr int IN_SIZE     = 12*64;
constexpr int HIDDEN_SIZE = 512;
constexpr int OUTPUT_SIZE = 1;

int main() {


    initLookUpTable();

    std::vector<Position> positions {};
    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\make_fens_from_lichess_pgn\output\0.txt)", &positions, P, 1);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\make_fens_from_lichess_pgn\output\1.txt)", &positions, P, 30000000);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\make_fens_from_lichess_pgn\output\2.txt)", &positions, P, 30000000);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\make_fens_from_lichess_pgn\output\3.txt)", &positions, P, 30000000);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\make_fens_from_lichess_pgn\output\4.txt)", &positions, P, 30000000);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\make_fens_from_lichess_pgn\output\5.txt)", &positions, P, 30000000);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\make_fens_from_lichess_pgn\output\6.txt)", &positions, P, 30000000);
//    write_positions_bin(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\eth\sfDump.bin)", &positions);
//    read_positions_bin(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\eth\sfDump.bin)", &positions);
    // creating buffers where inputs and targets will be stored for a batch
    std::vector<Input> inputs {};
    inputs.resize(BATCH_SIZE);

    std::vector<Data> targets {};
    for (int i = 0; i < BATCH_SIZE; i++) {
        targets.emplace_back(Data {OUTPUT_SIZE});
    }

    auto                         l1 = new DenseLayer<IN_SIZE    , HIDDEN_SIZE, ReLU> {};
    auto                         l2 = new DenseLayer<HIDDEN_SIZE, OUTPUT_SIZE, Linear> {};
    std::vector<LayerInterface*> layers {};
    layers.push_back(l1);
    layers.push_back(l2);

    Network network {layers};
    network.setLoss(new MSE());
    network.setOptimiser(new Adam());
//    network.loadWeights("fcSf.net");

//    Position p{};
//    p.set("r1bqk2r/ppppbppp/2n2n2/4p3/3PP3/2N2N2/PPP2PPP/R1BQKB1R w KQkq - 0 1;0.1");
//    PositionIterator it{p};
//    Input inp{};
//    Data output{1};
//    dense::assign_input(p, inp, output);
//
//    Data* h = network.evaluate(inp);
//    std::cout << *network.getLayer(1)->getWeights() << std::endl;
//    std::cout << *network.getThreadData(0)->output[0] << std::endl;
//    std::cout << *network.getThreadData(0)->output[1] << std::endl;
//    std::cout << *network.getLayer(1)->getBias() << std::endl;
//
//    std::cout << *h << std::endl;


    for (int i = 0; i < 100; i++) {

        int   batch_count = ceil(positions.size() / (float) BATCH_SIZE);

        float lossSum     = 0;

        auto  start       = std::chrono::system_clock::now();

        for (int batch = 0; batch < batch_count; batch++) {
            // fill the inputs and outputs
            int   batchsize = dense::assign_inputs_batch(positions, batch * BATCH_SIZE, inputs, targets);
            float loss      = network.batch(inputs, targets, batchsize, true);

            lossSum += loss * batchsize;
            auto                          end  = std::chrono::system_clock::now();
            std::chrono::duration<double> diff = end - start;


            printf("\repoch# %-10d batch# %-5d/%-10d loss=%-16.12f speed=%-7d eps", i, batch, batch_count, loss, (int) ((batch * BATCH_SIZE + batchsize) / diff.count()));
            std::cout << std::flush;
        }
        std::cout << std::endl;
//        std::cout << "train loss=" << lossSum / positions.size() << std::endl;

//        network.saveWeights("fcSf.net");
        network.newEpoch();
    }

    return 0;
}
