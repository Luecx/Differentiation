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

int nnue_index(Color active, Color kingColor, Color pieceColor, Square ksq, Square psq, PieceType p){

    const int relative_ksq = kingColor == WHITE ? ksq : mirror(ksq);
    const int relative_psq = kingColor == WHITE ? psq : mirror(psq);

    return
           (64 * 64 * 10 * (active != kingColor))  // skipping 64 * 64 * 10 if we use the opponents king
        +  (64 * 10      * relative_ksq)           // skipping 64 * 10 for each king square
        +  (64 * 5       * (active != pieceColor)) // skipping 64 * 5  if we are looking at the opponents pieces (e.g. wking/wpawn)
        +  (64           * p)
        + relative_psq;

}

void assign_input(Position &p, Input &input, Data& output){

    input.indices.clear();

    PositionIterator kingFinder{p};
    Square wksq = 0;
    Square bksq = 0;
    Color  active = kingFinder.activePlayer;
    int piece_count = 0;
    while(kingFinder.hasNext()){
        kingFinder.next();
        if(kingFinder.piece == WHITE_KING){
            wksq = kingFinder.sq;
        }else if(kingFinder.piece == BLACK_KING){
            bksq = kingFinder.sq;
        }else{
            piece_count ++;
        }
    }


    output(0) = kingFinder.score;
    input.indices.resize(2 * piece_count);
    PositionIterator it{p};
    int index = 0;
    while(it.hasNext()){
        it.next();
        if(it.piece == WHITE_KING || it.piece == BLACK_KING)
            continue;
        Color pieceColor = it.piece > WHITE_KING;

        if(active == WHITE){
            input.indices[0             + index] = nnue_index(active, WHITE, pieceColor, wksq, it.sq, it.piece % 6);
            input.indices[piece_count   + index] = nnue_index(active, BLACK, pieceColor, bksq, it.sq, it.piece % 6);
        }else{
            input.indices[piece_count   + index] = nnue_index(active, WHITE, pieceColor, wksq, it.sq, it.piece % 6);
            input.indices[0             + index] = nnue_index(active, BLACK, pieceColor, bksq, it.sq, it.piece % 6);
        }
        index ++;
    }
}

int assign_inputs_batch(std::vector<Position> &positions, int offset, std::vector<Input> &inputs, std::vector<Data> &targets){

    int end = offset + inputs.size();
    if(end > positions.size()) end = positions.size();

    int count = end - offset;

    for(int i = 0; i < count; i++){
        assign_input(positions[offset + i], inputs[i], targets[i]);
    }

    return count;
}

int main() {

    int BATCH_SIZE = 8192;

    initLookUpTable();

        std::vector<Position> positions{};
        read_positions_txt("F:\\OneDrive\\ProgrammSpeicher\\CLionProjects\\Koivisto\\resources\\other\\d8set.epd", &positions);

        std::vector<Input> inputs{};
        inputs.resize(BATCH_SIZE);

        std::vector<Data> targets{};
        for(int i = 0; i < BATCH_SIZE; i++){
            targets.emplace_back(Data{1});
        }




    //    write_positions_bin("F:\\OneDrive\\ProgrammSpeicher\\CLionProjects\\Koivisto\\resources\\other\\d8set.bin", &positions);

        constexpr int IN_SIZE = 64*10*64;
        constexpr int HIDDEN1_SIZE = 256;
        constexpr int HIDDEN2_SIZE = 32;
        constexpr int HIDDEN3_SIZE = 32;
        constexpr int OUTPUT_SIZE = 1;

        auto l1 = new DuplicateDenseLayer<IN_SIZE, HIDDEN1_SIZE, ReLU>{};
        auto l2 = new DenseLayer<HIDDEN1_SIZE*2, HIDDEN2_SIZE, ReLU>{};
        auto l3 = new DenseLayer<HIDDEN2_SIZE, HIDDEN3_SIZE, ReLU>{};
        auto l4 = new DenseLayer<HIDDEN3_SIZE, OUTPUT_SIZE, Linear>{};
        std::vector<LayerInterface *> layers{};
        layers.push_back(l1);
        layers.push_back(l2);
        layers.push_back(l3);
        layers.push_back(l4);

        Network network{layers};
        network.setLoss(new MSE());
        network.setOptimiser(new Adam());


        for(int i = 0; i < 300; i++){

            int batch_count = ceil(positions.size() / (float)BATCH_SIZE);

            float lossSum = 0;

            auto start = std::chrono::system_clock::now();

            for(int batch = 0; batch < batch_count; batch ++){
                // fill the inputs and outputs
                int batchsize = assign_inputs_batch(positions, batch * BATCH_SIZE, inputs, targets);
                float loss = network.batch(inputs, targets, batchsize);
                lossSum += loss * batchsize;
                auto end  = std::chrono::system_clock::now();
                std::chrono::duration<double> diff = end - start;
                printf("\repoch# %-10d batch# %-5d/%-10d loss=%-16.12f speed=%-7d eps",
                       i,batch,batch_count,loss,
                       (int)((batch * BATCH_SIZE + batchsize) / diff.count()));
                std::cout << std::flush;
            }
            std::cout << std::endl;
            std::cout << lossSum / positions.size() << std::endl;
            network.newEpoch();
        }


    return 0;

}
