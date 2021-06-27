
#include "Data.h"
#include "DenseLayer.h"
#include "Function.h"
#include "Input.h"
#include "Reader.h"
#include "mappings.h"
#include "network.h"
#include "optimiser.h"

#include <chrono>
int main() {

    constexpr int BATCH_SIZE  = 1024;

    constexpr int IN_SIZE     = 48 * 2;
    constexpr int HIDDEN_SIZE = 32;
    constexpr int OUTPUT_SIZE = 1;

    initLookUpTable();

    std::vector<Position> positions {};
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\eth\E12.33-1M-D12-Resolved.book)", &positions, 64);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\eth\E12.41-1M-D12-Resolved.book)", &positions, 30000000);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\eth\E12.52-1M-D12-Resolved.book)", &positions, 30000000);
//    read_positions_txt(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\eth\E12.46FRC-1250k-D12-1s.book)", &positions, 30000000);
//    write_positions_bin(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\eth\42.5M.bin)", &positions);
    read_positions_bin(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\eth\42.5M.bin)", &positions);

    // creating buffers where inputs and targets will be stored for a batch
    std::vector<Input> inputs {};
    inputs.resize(BATCH_SIZE);

    std::vector<Data> targets {};
    for (int i = 0; i < BATCH_SIZE; i++) {
        targets.emplace_back(Data {OUTPUT_SIZE});
    }

    auto                         l1 = new DenseLayer<IN_SIZE, HIDDEN_SIZE, Linear> {};
    auto                         l2 = new DenseLayer<HIDDEN_SIZE, OUTPUT_SIZE, Sigmoid> {};
    std::vector<LayerInterface*> layers {};
    layers.push_back(l1);
    layers.push_back(l2);

    Network network {layers};
    network.setLoss(new MSE());
    network.setOptimiser(new Adam());
    network.loadWeights("pawnNet1.net");

    Position p{};
    p.set("8/2pp2P1/3p1P2/8/3P4/2P5/8/8 b - - 0 1;0.1");
    PositionIterator it{p};
    std::cout << it.score << std::endl;
    Input inp{};
    Data output{1};
    dense_pawn::assign_input(p, inp, output);

    Data* h = network.evaluate(inp);
    std::cout << *h << std::endl;

//    for (int i = 0; i < 300; i++) {
//
//        int   batch_count = ceil(positions.size() / (float) BATCH_SIZE);
//
//        float lossSum     = 0;
//
//        auto  start       = std::chrono::system_clock::now();
//
//        for (int batch = 0; batch < batch_count; batch++) {
//            // fill the inputs and outputs
//            int   batchsize = dense_pawn::assign_inputs_batch(positions, batch * BATCH_SIZE, inputs, targets);
//            float loss      = network.batch(inputs, targets, batchsize);
//
//            lossSum += loss * batchsize;
//            auto                          end  = std::chrono::system_clock::now();
//            std::chrono::duration<double> diff = end - start;
//            printf("\repoch# %-10d batch# %-5d/%-10d loss=%-16.12f speed=%-7d eps", i, batch, batch_count, loss, (int) ((batch * BATCH_SIZE + batchsize) / diff.count()));
//            std::cout << std::flush;
//        }
//        std::cout << std::endl;
//        std::cout << "train loss=" << lossSum / positions.size() << std::endl;
//
//        network.saveWeights("pawnNet1.net");
//        network.newEpoch();
//    }

    return 0;
}
