//
// Created by Luecx on 03.03.2021.
//

#ifndef DIFFERENTIATION_MAPPINGS_H
#define DIFFERENTIATION_MAPPINGS_H

#include "Data.h"
#include "Reader.h"

namespace halfkp {

inline int index(Color active, Color kingColor, Color pieceColor, Square ksq, Square psq, PieceType p) {

    const int relative_ksq = kingColor == WHITE ? ksq : mirror(ksq);
    const int relative_psq = kingColor == WHITE ? psq : mirror(psq);

    return ((64 * 64 * 10 + 2250) * (active != kingColor))    // skipping 64 * 64 * 10 if we use the opponents king + the factor
           + (64 * 10 * relative_ksq)                         // skipping 64 * 10 for each king square
           + (64 * 5 * (active != pieceColor))                // skipping 64 * 5  if we are looking at the opponents pieces (e.g. wking/wpawn)
           + (64 * p) + relative_psq;
}

inline int index_factor_1(Color active, Color kingColor, Color pieceColor, Square ksq, Square psq, PieceType p) {

    const int relative_ksq             = kingColor == WHITE ? ksq : mirror(ksq);
    const int relative_psq             = kingColor == WHITE ? psq : mirror(psq);

    const int relative_rank_difference = rankIndex(relative_ksq) - rankIndex(relative_psq);
    const int relative_file_difference = fileIndex(relative_ksq) - fileIndex(relative_psq);

    return ((64 * 64 * 10 + 2250) * (active != kingColor))    // skipping 64 * 64 * 10 if we use the opponents king + the factor
           + (64 * 64 * 10                                    // skipping 64 * 64 * 10 for the nnue_index
              + (15 * 15 * 5 * (active != pieceColor))        // skipping 15 * 15 * 5  if we are looking at the opponents pieces (e.g. wking/bpawn)
              + (15 * 15 * p)                                 // skipping 15 * 15      to reserve for each piece type
              + ((relative_rank_difference + 7) * 15 + (relative_file_difference + 7)));
}

inline void assign_input(Position& p, Input& input, Data& output, bool original = false) {

    input.indices.clear();

    PositionIterator kingFinder {p};
    Square           wksq        = 0;
    Square           bksq        = 0;
    Color            active      = kingFinder.activePlayer;
    int              piece_count = 0;
    while (kingFinder.hasNext()) {
        kingFinder.next();
        if (kingFinder.piece == WHITE_KING) {
            wksq = kingFinder.sq;
        } else if (kingFinder.piece == BLACK_KING) {
            bksq = kingFinder.sq;
        } else {
            piece_count++;
        }
    }

    kingFinder.score *= active == WHITE ? 1 : -1;
    output(0) = std::max((short) -6, std::min((short) 6, kingFinder.score));

    if (original) {
        input.indices.resize(2 * piece_count);
    } else {
        input.indices.resize(4 * piece_count);
    }
    PositionIterator it {p};
    int              index = 0;
    while (it.hasNext()) {
        it.next();
        if (it.piece == WHITE_KING || it.piece == BLACK_KING)
            continue;
        Color pieceColor = it.piece > WHITE_KING;

        if (active == WHITE) {
            if (!original) {
                input.indices[0 + index]               = halfkp::index(active, WHITE, pieceColor, wksq, it.sq, it.piece % 6);
                input.indices[piece_count * 2 + index] = halfkp::index(active, BLACK, pieceColor, bksq, it.sq, it.piece % 6);
                input.indices[piece_count * 1 + index] = halfkp::index_factor_1(active, WHITE, pieceColor, wksq, it.sq, it.piece % 6);
                input.indices[piece_count * 3 + index] = halfkp::index_factor_1(active, BLACK, pieceColor, bksq, it.sq, it.piece % 6);
            } else {
                input.indices[0 + index]               = halfkp::index(active, WHITE, pieceColor, wksq, it.sq, it.piece % 6);
                input.indices[piece_count * 1 + index] = halfkp::index(active, BLACK, pieceColor, bksq, it.sq, it.piece % 6);
            }
        } else {
            if (!original) {
                input.indices[piece_count * 2 + index] = halfkp::index(active, WHITE, pieceColor, wksq, it.sq, it.piece % 6);
                input.indices[0 + index]               = halfkp::index(active, BLACK, pieceColor, bksq, it.sq, it.piece % 6);
                input.indices[piece_count * 3 + index] = halfkp::index_factor_1(active, WHITE, pieceColor, wksq, it.sq, it.piece % 6);
                input.indices[piece_count * 1 + index] = halfkp::index_factor_1(active, BLACK, pieceColor, bksq, it.sq, it.piece % 6);
            } else {
                input.indices[piece_count * 1 + index] = halfkp::index(active, WHITE, pieceColor, wksq, it.sq, it.piece % 6);
                input.indices[0 + index]               = halfkp::index(active, BLACK, pieceColor, bksq, it.sq, it.piece % 6);
            }
        }
        index++;
    }
}

inline int assign_inputs_batch(std::vector<Position>& positions, int offset, std::vector<Input>& inputs, std::vector<Data>& targets) {

    int end = offset + inputs.size();
    if (end > positions.size())
        end = positions.size();

    int count = end - offset;

    for (int i = 0; i < count; i++) {
        assign_input(positions[offset + i], inputs[i], targets[i]);
    }

    return count;
}

inline void reduce_matrix(Data& original, Data& reduced) {
    // we can copy the first 64 * 10 * 64 columns
    for (int i = 0; i < 64 * 10 * 64; i++) {

        const Square relative_ksq             = i / (64 * 10);
        const Piece  relativePiece            = (i % (64 * 10)) / 64;
        const Piece  relative_psq             = (i % 64);

        const Rank   relative_rank_difference = rankIndex(relative_ksq) - rankIndex(relative_psq);
        const Rank   relative_file_difference = fileIndex(relative_ksq) - fileIndex(relative_psq);

        const int    factor_offset            = 64 * 64 * 10;
        const int    factor_1_offset          = factor_offset + 15 * 15 * relativePiece + ((relative_rank_difference + 7) * 15 + (relative_file_difference + 7));

        for (int k = 0; k < original.getM(); k++) {
            reduced.values[i * original.getM() + k] = original.values[i * original.getM() + k] + original.values[factor_1_offset * original.getM() + k];
        }
    }
}

};    // namespace halfkp

namespace ataxx {

struct Sample {
    Sample() : activePieces {}, passivePieces {}, target {} {}

    void set(const std::string& fen) {

        std::string       word;

        std::stringstream ss {fen};
        std::uint64_t     mask_x = 0ULL;
        std::uint64_t     mask_o = 0ULL;

        // Eval target
        {
            ss >> word;
            target = std::stof(word);
            assert(std::abs(target) < 9000);
        }

        // Pieces
        {
            ss >> word;
            int idx = 0;
            for (const auto& c : word) {
                switch (c) {
                    case 'x':
                        mask_x ^= (1ULL << idx);
                        idx++;
                        break;
                    case 'o':
                        mask_o ^= (1ULL << idx);
                        idx++;
                        break;
                    case '-': idx++; break;
                    case '1':
                    case '2':
                    case '3':
                    case '4':
                    case '5':
                    case '6':
                    case '7': idx += c - '0'; break;
                    case '/': break;
                    default: break;
                }
            }

            assert(idx == 49);
            assert(!(mask_x & mask_o));
        }

        // Side to move
        ss >> word;
        while (mask_x) {
            const int sq = __builtin_ctzll(mask_x);
            if (word == "x") {
                activePieces.push_back(sq);
            } else {
                passivePieces.push_back(sq);
            }
            mask_x &= mask_x - 1;
        }
        while (mask_o) {
            const int sq = __builtin_ctzll(mask_o);
            if (word == "o") {
                activePieces.push_back(sq);
            } else {
                passivePieces.push_back(sq);
            }
            mask_o &= mask_o - 1;
        }
    }

    std::vector<std::uint8_t> activePieces;
    std::vector<std::uint8_t> passivePieces;
    float                     target;
};


inline void load_positions(const std::string& fil, std::vector<ataxx::Sample>& samples) {
    std::string   line;
    std::string   word;
    std::ifstream file(fil);

    while (std::getline(file, line)) {
        if (samples.size() % 1000 == 0) {
            printf("\r[Loading positions] Current size=%d", samples.size());
            fflush(stdout);
        }

        std::stringstream ss {line};
        ataxx::Sample     s;
        std::uint64_t     mask_x = 0ULL;
        std::uint64_t     mask_o = 0ULL;

        // Eval target
        {
            ss >> word;
            s.target = std::stof(word);

            assert(std::abs(s.target) < 9000);
        }
        if (abs(s.target) > 800)
            continue;

        // Pieces
        {
            ss >> word;
            int idx = 0;
            for (const auto& c : word) {
                switch (c) {
                    case 'x':
                        mask_x ^= (1ULL << idx);
                        idx++;
                        break;
                    case 'o':
                        mask_o ^= (1ULL << idx);
                        idx++;
                        break;
                    case '-': idx++; break;
                    case '1':
                    case '2':
                    case '3':
                    case '4':
                    case '5':
                    case '6':
                    case '7': idx += c - '0'; break;
                    case '/': break;
                    default: break;
                }
            }

            assert(idx == 49);
            assert(!(mask_x & mask_o));
        }

        // Side to move
        ss >> word;
        while (mask_x) {
            const int sq = __builtin_ctzll(mask_x);
            if (word == "x") {
                s.activePieces.push_back(sq);
            } else {
                s.passivePieces.push_back(sq);
            }
            mask_x &= mask_x - 1;
        }
        while (mask_o) {
            const int sq = __builtin_ctzll(mask_o);
            if (word == "o") {
                s.activePieces.push_back(sq);
            } else {
                s.passivePieces.push_back(sq);
            }
            mask_o &= mask_o - 1;
        }
        samples.push_back(s);

        // process pair (a,b)
    }

    std::cout << std::endl;
}

inline void assign_input(ataxx::Sample& p, Input& input, Data& output) {

    input.indices.clear();

    input.indices.reserve(p.activePieces.size() + p.passivePieces.size());
    for (uint8_t t : p.activePieces) {
        input.indices.push_back(t);
    }
    for (uint8_t t : p.passivePieces) {
        input.indices.push_back(49 + t);
    }

    output(0) = p.target;
}

inline int assign_inputs_batch(std::vector<Sample>& positions, int offset, std::vector<Input>& inputs, std::vector<Data>& targets) {

    int end = offset + inputs.size();
    if (end > positions.size())
        end = positions.size();

    int count = end - offset;

    for (int i = 0; i < count; i++) {
        assign_input(positions[offset + i], inputs[i], targets[i]);
    }

    return count;
}



}    // namespace ataxx

namespace dense{

inline int  index(Square psq, Piece p, Color activePlayer) {

    Square    relativeSquare = psq;
    Color     pieceColor     = p >= BLACK_PAWN ? BLACK : WHITE;
    PieceType pieceType      = p % 6;

    return relativeSquare + pieceColor * 64 * 6 + pieceType * 64;
}

inline void assign_input(Position& p, Input& input, Data& output){
    PositionIterator it {p};
    output(0) = it.score;
    input.indices.clear();
    while (it.hasNext()) {
        it.next();
        input.indices.push_back(index(it.sq, it.piece, it.activePlayer));
    }
}

inline int  assign_inputs_batch(std::vector<Position>& positions, int offset, std::vector<Input>& inputs, std::vector<Data>& targets){
    int end = offset + inputs.size();
    if (end > positions.size())
        end = positions.size();

    int count = end - offset;

    for (int i = 0; i < count; i++) {
        assign_input(positions[offset + i], inputs[i], targets[i]);
    }

    return count;
}
}

namespace dense_relative {

inline int  index(Square psq, Piece p, Color activePlayer) {

//    Square    relativeSquare = psq;
//    Color     pieceColor     = p >= BLACK_PAWN ? BLACK : WHITE;
//    PieceType pieceType      = p % 6;
//
//    return relativeSquare + pieceColor * 64 * 6 + pieceType * 64;

    Square    relativeSquare = activePlayer == WHITE ? psq : mirror(psq);
    Color     pieceColor     = p >= BLACK_PAWN ? BLACK : WHITE;
    PieceType pieceType      = p % 6;

    return relativeSquare + (pieceColor == activePlayer) * 64 * 6 + pieceType * 64;

}

inline void assign_input(Position& p, Input& input, Data& output){
    PositionIterator it {p};
//    output(0) = 1 / (1 + exp(-it.score));
    output(0) = it.activePlayer == WHITE ? it.score : -it.score;
//    output(0) = 1 / (1 + exp(-it.score * SIGMOID_SCALE));
    input.indices.clear();
    while (it.hasNext()) {
        it.next();
        input.indices.push_back(index(it.sq, it.piece, it.activePlayer));
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

namespace dense_pawn {

inline int index(Square psq, Piece p, Color sideToMove) {

    Square relativeSquare = sideToMove == WHITE ? psq : mirror(psq);
    Color  pieceColor     = p >= BLACK_PAWN ? BLACK : WHITE;
    int    offset         = (sideToMove == pieceColor) * 48;

    return relativeSquare - 8 + offset;
}

inline void assign_input(Position& p, Input& input, Data& output) {
    PositionIterator it {p};
    output(0) = 1 / (1 + exp(-it.score * (it.activePlayer == WHITE ? 1 : -1)));
    input.indices.clear();

    while (it.hasNext()) {
        it.next();

        if (it.piece != WHITE_PAWN && it.piece != BLACK_PAWN)
            continue;

        input.indices.push_back(index(it.sq, it.piece, it.activePlayer));
    }
}

inline int assign_inputs_batch(std::vector<Position>& positions, int offset, std::vector<Input>& inputs, std::vector<Data>& targets) {
    int end = offset + inputs.size();
    if (end > positions.size())
        end = positions.size();

    int count = end - offset;

    for (int i = 0; i < count; i++) {
        assign_input(positions[offset + i], inputs[i], targets[i]);
    }

    return count;
}
}    // namespace dense_pawn

namespace dense_relative_extended {
inline int  index(Square psq, Piece p, Color activePlayer) {

//    Square    relativeSquare = psq;
//    Color     pieceColor     = p >= BLACK_PAWN ? BLACK : WHITE;
//    PieceType pieceType      = p % 6;
//
//    return relativeSquare + pieceColor * 64 * 6 + pieceType * 64;

    Square    relativeSquare = activePlayer == WHITE ? psq : mirror(psq);
    Color     pieceColor     = p >= BLACK_PAWN ? BLACK : WHITE;
    PieceType pieceType      = p % 6;

    return relativeSquare + (pieceColor == activePlayer) * 64 * 6 + pieceType * 64;

}

inline void assign_input(Position& p, Input& input, Data& output){
    PositionIterator it {p};

    output(0) = 1 / (1 + exp(-it.score * SIGMOID_SCALE));
//    output(0) = it.activePlayer == WHITE ? it.score : -it.score;
    input.indices.clear();
    while (it.hasNext()) {
        it.next();
        input.indices.push_back(index(it.sq, it.piece, it.activePlayer));
    }
}

inline int  assign_inputs_batch(std::vector<Position>& positions, int offset, std::vector<Input>& inputs, std::vector<Data>& targets){
    int end = offset + inputs.size();
    if (end > positions.size())
        end = positions.size();

    int count = end - offset;

    for (int i = 0; i < count; i++) {
        assign_input(positions[offset + i], inputs[i], targets[i]);
    }

    return count;
}
}

#endif    // DIFFERENTIATION_MAPPINGS_H
