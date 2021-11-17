//
// Created by Luecx on 03.03.2021.
//

#ifndef DIFFERENTIATION_MAPPINGS_H
#define DIFFERENTIATION_MAPPINGS_H

#include "misc/Reader.h"
#include "structures/Data.h"

namespace dense_relative {

inline int  index(Square psq, Piece p, Square kingSquare, Color view) {

    Square    relativeSquare = view == WHITE ? psq : mirror(psq);
    Color     pieceColor     = p >= BLACK_PAWN ? BLACK : WHITE;
    PieceType pieceType      = p % 6;
    bool      kingSide       = (kingSquare & 7) > 3;

    if (kingSide){
        relativeSquare ^= 7;
    }

    return relativeSquare
           + (pieceColor == view) * 64 * 6
           + pieceType * 64;

}

inline void assign_input(Position& p, Input& input, Data& output){
    PositionIterator it {p};

    int16_t     tar       = it.activePlayer == WHITE ? it.score : -it.score;
    float       WDLtarget = 0.5;
    if (tar > 10000) {
        WDLtarget = 1;
        tar -= 20000;
    }
    if (tar < -10000) {
        WDLtarget = 0;
        tar += 20000;
    }
    float Ptarget = 1 / (1 + expf(-tar * SIGMOID_SCALE));
    output(0) = (WDLtarget + Ptarget) / 2;



    // track king squares
    Square wKingSq = 0;
    Square bKingSq = 0;
    while (it.hasNext()) {
        it.next();
        if(it.piece == WHITE_KING)
            wKingSq = it.sq;
        if(it.piece == BLACK_KING)
            bKingSq = it.sq;
    }

    // read all the pieces
    input.indices.clear();
    it = PositionIterator{p};
    while (it.hasNext()) {
        it.next();

        auto piece_index_white_pov = index(it.sq, it.piece, wKingSq, WHITE);
        auto piece_index_black_pov = index(it.sq, it.piece, bKingSq, BLACK);

        if(it.activePlayer == WHITE){
            piece_index_black_pov += 12 * 64;
        }else{
            piece_index_white_pov += 12 * 64;
        }

        input.indices.push_back(piece_index_white_pov);
        input.indices.push_back(piece_index_black_pov);
    }

}

inline int  assign_inputs_batch(std::vector<Position>& positions, int offset, std::vector<Input>& inputs, std::vector<Data>& targets){
    int end = offset + inputs.size();
    if (end > positions.size())
        end = positions.size();

    int count = end - offset;

#pragma omp parallel for schedule(auto) num_threads(UPDATE_THREADS)
    for (int i = 0; i < count; i++) {
        assign_input(positions[offset + i], inputs[i], targets[i]);
    }

    return count;
}
}    // namespace dense


#endif    // DIFFERENTIATION_MAPPINGS_H
