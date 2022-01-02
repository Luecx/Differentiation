//
// Created by Luecx on 03.03.2021.
//

#ifndef DIFFERENTIATION_MAPPINGS_H
#define DIFFERENTIATION_MAPPINGS_H

#include "dataset/dataset.h"
#include "position/fenparsing.h"
#include "structures/Data.h"

#include <cmath>

namespace dense_relative {

inline int index(Square psq, Piece p, Square kingSquare, Color view) {

    Square    relativeSquare = view == WHITE ? psq : mirrorVertically(psq);
    Color     pieceColor     = p >= BLACK_PAWN ? BLACK : WHITE;
    PieceType pieceType      = p % 6;
    bool      kingSide       = (kingSquare & 7) > 3;

    if (kingSide) {
        relativeSquare ^= 7;
    }

    return relativeSquare + (pieceColor == view) * 64 * 6 + pieceType * 64;
}

inline void assign_input(Position& p, Input& input, Data& output) {


    float p_value  = p.m_result.score;
    float w_value  = p.m_result.wdl;

    // flip if black is to move -> relative network style
    if(p.m_meta.getActivePlayer() == BLACK){
        p_value = -p_value;
        w_value = -w_value;
    }

    float p_target = 1 / (1 + expf(-p_value * SIGMOID_SCALE));
    float w_target = (w_value + 1) / 2.0f;

    output(0) = (p_target + w_target) / 2;

    // track king squares
    Square wKingSq = p.getKingSquare<WHITE>();
    Square bKingSq = p.getKingSquare<BLACK>();

    input.indices.clear();
    BB bb{p.m_occupancy};
    int idx = 0;

    while(bb){
        Square sq  = bitscanForward(bb);
        Piece  pc = p.m_pieces.getPiece(idx);

        auto piece_index_white_pov = index(sq, pc, wKingSq, WHITE);
        auto piece_index_black_pov = index(sq, pc, bKingSq, BLACK);

        if (p.m_meta.getActivePlayer() == WHITE) {
            piece_index_black_pov += 12 * 64;
        } else {
            piece_index_white_pov += 12 * 64;
        }

        input.indices.push_back(piece_index_white_pov);
        input.indices.push_back(piece_index_black_pov);

        bb = lsbReset(bb);
        idx ++;
    }

//    // read all the pieces
//    input.indices.clear();
//    for(int i = 0; i < p.getPieceCount(); i++){
//
//        auto piece_index_white_pov = index(p.getSquare(i), p.m_pieces.getPiece(i), wKingSq, WHITE);
//        auto piece_index_black_pov = index(p.getSquare(i), p.m_pieces.getPiece(i), bKingSq, BLACK);
//
//        if (p.m_meta.getActivePlayer() == WHITE) {
//            piece_index_black_pov += 12 * 64;
//        } else {
//            piece_index_white_pov += 12 * 64;
//        }
//
//        input.indices.push_back(piece_index_white_pov);
//        input.indices.push_back(piece_index_black_pov);
//    }

}

inline int  assign_inputs_batch(DataSet& positions, std::vector<Input>& inputs, std::vector<Data>& targets, int offset=0) {

    auto size = std::min(positions.header.position_count - offset, inputs.size());

#pragma omp parallel for schedule(static) num_threads(2)
    for (int i = 0; i < size; i++) {
        assign_input(positions.positions[i + offset], inputs[i], targets[i]);
    }

    return size;
}
}    // namespace dense_relative

#endif    // DIFFERENTIATION_MAPPINGS_H
