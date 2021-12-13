
#include "activations/ReLU.h"
#include "activations/Sigmoid.h"
#include "dataset/batchloader.h"
#include "dataset/reader.h"
#include "dataset/writer.h"
#include "layers/DenseLayer.h"
#include "layers/DuplicateDenseLayer.h"
#include "loss/MSE.h"
#include "network/network.h"
#include "optimiser/Adam.h"
#include "optimiser/optimiser.h"
#include "structures/Data.h"
#include "structures/Input.h"
#include "verify/checkGradients.h"

#include <algorithm>
#include <chrono>
#include <random>



int batch_size        = 1<<14;
int batches_in_memory = 256;
int input_size        = 768;
int batch_memory      = batch_size * input_size;
std::vector<std::string> sources{};
BatchLoader* batch_loader;

float*   batch_stm    = nullptr;
float*   batch_nstm   = nullptr;
float*   cp_values    = nullptr;
float*   wdl_values   = nullptr;

int  mappingIndex(Square psq, Piece p, Square kingSquare, Color view) {
    Color     pieceColor     = p >= BLACK_PAWN ? BLACK : WHITE;
    PieceType pieceType      = p % 6;
    bool      kingSide       = (kingSquare & 7) > 3;

    constexpr int pieceTypeFactor  = 64;
    constexpr int pieceColorFactor = 64 * 6;
    constexpr int kingSideFactor   = 64 * 6 * 2;

    Square  relativeSquare   = view == WHITE ? psq : mirrorVertically(psq);

    return relativeSquare
        + pieceType * pieceTypeFactor
        + (pieceColor == view) * pieceColorFactor
        + kingSide * kingSideFactor;
}
void mapPosition(Position& p, float* stm, float* nstm, float* wdl, float* cp) {


    float p_value  = p.m_result.score;
    float w_value  = p.m_result.wdl;

    // flip if black is to move -> relative network style
    if(p.m_meta.getActivePlayer() == BLACK){
        p_value = -p_value;
        w_value = -w_value;
    }

    *wdl = w_value;
    * cp = p_value;

    // track king squares
    Square wKingSq = p.getKingSquare<WHITE>();
    Square bKingSq = p.getKingSquare<BLACK>();

    // read all the pieces
    for(int i = 0; i < p.getPieceCount(); i++){
        auto piece_index_white_pov = mappingIndex(p.getSquare(i), p.m_pieces.getPiece(i), wKingSq, WHITE);
        auto piece_index_black_pov = mappingIndex(p.getSquare(i), p.m_pieces.getPiece(i), bKingSq, BLACK);

        if (p.m_meta.getActivePlayer() == WHITE) {
            stm [piece_index_white_pov] = 1;
            nstm[piece_index_black_pov] = 1;
        } else {
            nstm[piece_index_white_pov] = 1;
            stm [piece_index_black_pov] = 1;
        }

    }
}
void map(DataSet* positions) {
#pragma omp parallel for schedule(auto) num_threads(4)
    for (int i = 0; i < positions->header.position_count; i++) {
        mapPosition(positions->positions[i],
                    &batch_stm [input_size * i],
                    &batch_nstm[input_size * i],
                    &wdl_values[i],
                    &cp_values [i]);
    }
}    // namespace dense_relative

void initBatches(int p_batch_size, int p_batches_in_memory){
    batch_size        = p_batch_size;
    batches_in_memory = p_batches_in_memory;
    batch_memory      = batch_size * input_size;
}
void addSource(const std::string& file_name){
    sources.push_back(file_name);
}
void begin(){
    batch_loader = new BatchLoader(sources, batch_size, batches_in_memory);

    // init batch
    batch_stm  = new float[batch_memory] {};
    batch_nstm = new float[batch_memory] {};
    cp_values  = new float[batch_size] {};
    wdl_values = new float[batch_size] {};
}
void end(){
    delete batch_loader;
    delete[] batch_stm;
    delete[] batch_nstm;
    delete[] cp_values;
    delete[] wdl_values;
}
void next(){
    std::memset(batch_stm , 0, batch_memory);
    std::memset(batch_nstm, 0, batch_memory);

    DataSet* ds = batch_loader->next();
    map(ds);
}

float* batchSTMMemory() { return batch_stm; }
float* batchNSTMMemory() { return batch_nstm; }
float* cpMemory() {
    return cp_values;
}
float* wdlMemory() {
    return wdl_values;
}


int main() {

    BB bb = 1231823718;
    printBitboard(bb);
    std::cout << (int)bitscanForwardIndex(bb, 3);


//    initBatches(16348, 16);
//    addSource(R"(F:\OneDrive\ProgrammSpeicher\CLionProjects\Koivisto\resources\tuningsets\koi\koi7.9\generated_0.txt.bin)");
//
//    begin();
//
//    next();

//    end();

    return 0;
}
