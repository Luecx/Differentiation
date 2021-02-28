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

int nnue_index(Color active, Color kingColor, Color pieceColor, Square ksq, Square psq, PieceType p){

    const int relative_ksq = kingColor == WHITE ? ksq : mirror(ksq);
    const int relative_psq = kingColor == WHITE ? psq : mirror(psq);

    return
           ((64 * 64 * 10 + 2250) * (active != kingColor))  // skipping 64 * 64 * 10 if we use the opponents king + the factor
        +  ( 64 * 10              * relative_ksq)           // skipping 64 * 10 for each king square
        +  ( 64 * 5               * (active != pieceColor)) // skipping 64 * 5  if we are looking at the opponents pieces (e.g. wking/wpawn)
        +  ( 64                   * p)
        + relative_psq;

}

int nnue_index_factor_1(Color active, Color kingColor, Color pieceColor, Square ksq, Square psq, PieceType p){

    const int relative_ksq = kingColor == WHITE ? ksq : mirror(ksq);
    const int relative_psq = kingColor == WHITE ? psq : mirror(psq);

    const int relative_rank_difference = rankIndex(relative_ksq) - rankIndex(relative_psq);
    const int relative_file_difference = fileIndex(relative_ksq) - fileIndex(relative_psq);

    return
               ((64 * 64 * 10 + 2250) * (active != kingColor))  // skipping 64 * 64 * 10 if we use the opponents king + the factor
            +   (64 * 64 * 10                                   // skipping 64 * 64 * 10 for the nnue_index
            +   (15 * 15 * 5          * (active != pieceColor)) // skipping 15 * 15 * 5  if we are looking at the opponents pieces (e.g. wking/bpawn)
            +   (15 * 15              * p)                      // skipping 15 * 15      to reserve for each piece type
            +   ((relative_rank_difference + 7) * 15 + (relative_file_difference + 7)));
}

void assign_input(Position &p, Input &input, Data& output, bool original=false){

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

    kingFinder.score *= active == WHITE ? 1:-1;
    output(0) = std::max((short)-6, std::min((short)6,kingFinder.score));

    if(original){
        input.indices.resize(2 * piece_count);
    }else{
        input.indices.resize(4 * piece_count);
    }
    PositionIterator it{p};
    int index = 0;
    while(it.hasNext()){
        it.next();
        if(it.piece == WHITE_KING || it.piece == BLACK_KING)
            continue;
        Color pieceColor = it.piece > WHITE_KING;

        if(active == WHITE){
            if(!original){
                input.indices[0               + index] = nnue_index         (active, WHITE, pieceColor, wksq, it.sq, it.piece % 6);
                input.indices[piece_count*2   + index] = nnue_index         (active, BLACK, pieceColor, bksq, it.sq, it.piece % 6);
                input.indices[piece_count*1   + index] = nnue_index_factor_1(active, WHITE, pieceColor, wksq, it.sq, it.piece % 6);
                input.indices[piece_count*3   + index] = nnue_index_factor_1(active, BLACK, pieceColor, bksq, it.sq, it.piece % 6);
            }else{
                input.indices[0               + index] = nnue_index         (active, WHITE, pieceColor, wksq, it.sq, it.piece % 6);
                input.indices[piece_count*1   + index] = nnue_index         (active, BLACK, pieceColor, bksq, it.sq, it.piece % 6);
            }
        }else{
            if(!original){
                input.indices[piece_count*2   + index] = nnue_index         (active, WHITE, pieceColor, wksq, it.sq, it.piece % 6);
                input.indices[0               + index] = nnue_index         (active, BLACK, pieceColor, bksq, it.sq, it.piece % 6);
                input.indices[piece_count*3   + index] = nnue_index_factor_1(active, WHITE, pieceColor, wksq, it.sq, it.piece % 6);
                input.indices[piece_count*1   + index] = nnue_index_factor_1(active, BLACK, pieceColor, bksq, it.sq, it.piece % 6);
            }else{
                input.indices[piece_count*1   + index] = nnue_index         (active, WHITE, pieceColor, wksq, it.sq, it.piece % 6);
                input.indices[0               + index] = nnue_index         (active, BLACK, pieceColor, bksq, it.sq, it.piece % 6);
            }
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

void nnue_reduce_matrix(Data& original, Data& reduced){
    // we can copy the first 64 * 10 * 64 columns
    for(int i = 0; i < 64 * 10 * 64; i++){

        const Square relative_ksq =  i / (64 * 10);
        const Piece relativePiece = (i % (64 * 10)) / 64;
        const Piece relative_psq  = (i % 64);

        const Rank relative_rank_difference = rankIndex(relative_ksq) - rankIndex(relative_psq);
        const Rank relative_file_difference = fileIndex(relative_ksq) - fileIndex(relative_psq);

        const int factor_offset   = 64 * 64 * 10;
        const int factor_1_offset = factor_offset +
                15 * 15 * relativePiece +
                ((relative_rank_difference + 7) * 15 + (relative_file_difference + 7));

        for(int k = 0; k < original.getM(); k++){
            reduced.values[i * original.getM() + k] =
                     original.values[i                 * original.getM() + k] +
                     original.values[factor_1_offset   * original.getM() + k];

        }

    }
}

int main() {

    int BATCH_SIZE = 8192;

    initLookUpTable();

//    // loading positions
//    std::vector<Position> positions{};
//    positions.reserve(300 * 1000 * 1000);
//    read_positions_bin("position.bin",&positions);
//    std::cout << "beginning shuffling" << std::endl;
//    std::shuffle(positions.begin(), positions.end(), std::random_device());
//    std::cout << "finished shuffling" << std::endl;

    // creating buffers where inputs and targets will be stored for a batch
    std::vector<Input> inputs{};
    inputs.resize(BATCH_SIZE);

    std::vector<Data> targets{};
    for (int i = 0; i < BATCH_SIZE; i++) {
        targets.emplace_back(Data{1});
    }

    // setting up the network
    constexpr int IN_SIZE = 64 * 10 * 64 + 2250;
    constexpr int HIDDEN1_SIZE = 256;
    constexpr int HIDDEN2_SIZE = 32;
    constexpr int HIDDEN3_SIZE = 32;
    constexpr int OUTPUT_SIZE = 1;

    auto l1 = new DuplicateDenseLayer<IN_SIZE, HIDDEN1_SIZE, ReLU>{};
    auto l2 = new DenseLayer<HIDDEN1_SIZE * 2, HIDDEN2_SIZE, ReLU>{};
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

    network.loadWeights("test.net");

    Position    test_pos{};
    Input       test_pos_in{};
    Data        test_pos_tar{1};
    test_pos.set("rnbqkb1r/pppppppp/8/3n4/8/4P3/PPPP1PPP/RNBQKBNR w KQq - 0 1");
    assign_input(test_pos, test_pos_in, test_pos_tar);
    std::cout << *network.evaluate(test_pos_in) << std::endl;

//    for (int i = 0; i < 300; i++) {
//
//        int batch_count = ceil(positions.size() / (float) BATCH_SIZE);
//
//        float lossSum = 0;
//
//        auto start = std::chrono::system_clock::now();
//
//        for (int batch = 0; batch < batch_count; batch++) {
//            // fill the inputs and outputs
//            int batchsize = assign_inputs_batch(positions, batch * BATCH_SIZE, inputs, targets);
//            float loss = network.batch(inputs, targets, batchsize);
//            lossSum += loss * batchsize;
//            auto end = std::chrono::system_clock::now();
//            std::chrono::duration<double> diff = end - start;
//            printf("\repoch# %-10d batch# %-5d/%-10d loss=%-16.12f speed=%-7d eps",
//                   i, batch, batch_count, loss,
//                   (int) ((batch * BATCH_SIZE + batchsize) / diff.count()));
//            std::cout << std::flush;
//        }
//        std::cout << std::endl;
//        std::cout << lossSum / positions.size() << std::endl;
//        network.newEpoch();
//        network.saveWeights("test.net");
//    }


    return 0;

}
