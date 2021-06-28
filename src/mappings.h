//
// Created by Luecx on 03.03.2021.
//

#ifndef DIFFERENTIATION_MAPPINGS_H
#define DIFFERENTIATION_MAPPINGS_H

#include "Data.h"
#include "Reader.h"

namespace nnue {
int  index(Color active, Color kingColor, Color pieceColor, Square ksq, Square psq, PieceType p);

int  index_factor_1(Color active, Color kingColor, Color pieceColor, Square ksq, Square psq, PieceType p);

void assign_input(Position& p, Input& input, Data& output, bool original = false);

int  assign_inputs_batch(std::vector<Position>& positions, int offset, std::vector<Input>& inputs, std::vector<Data>& targets);

void reduce_matrix(Data& original, Data& reduced);
};    // namespace nnue

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

void load_positions(const std::string& fil, std::vector<Sample>& samples);

void assign_input(Sample& p, Input& input, Data& output);

int  assign_inputs_batch(std::vector<Sample>& positions, int offset, std::vector<Input>& inputs, std::vector<Data>& targets);

}    // namespace ataxx

namespace dense {
int  index(Square psq, Piece p, Color activePlayer);

//    int index_factor_1(Color active, Color kingColor, Color pieceColor, Square ksq, Square psq, PieceType p);

void assign_input(Position& p, Input& input, Data& output);

int  assign_inputs_batch(std::vector<Position>& positions, int offset, std::vector<Input>& inputs, std::vector<Data>& targets);
}    // namespace dense

namespace dense_pawn {


inline int index(Square psq, Piece p, Color sideToMove) {

    Square relativeSquare = sideToMove == WHITE ? psq : mirror(psq);
    Color  pieceColor     = p >= BLACK_PAWN ? BLACK : WHITE;
    int    offset         = (sideToMove == pieceColor) * 48;

    return relativeSquare - 8 + offset;
}

inline void assign_input(Position &p, Input &input, Data &output) {
    PositionIterator it{p};
    output(0) = 1 / (1 + exp(-it.score * (it.activePlayer == WHITE ? 1:-1)));
    input.indices.clear();

    while(it.hasNext()){
        it.next();

        if(it.piece != WHITE_PAWN && it.piece != BLACK_PAWN) continue;

        input.indices.push_back(index(it.sq, it.piece, it.activePlayer));
    }
}

inline int assign_inputs_batch(std::vector<Position> &positions, int offset, std::vector<Input> &inputs,
                               std::vector<Data> &targets) {
    int end = offset + inputs.size();
    if(end > positions.size()) end = positions.size();

    int count = end - offset;

    for(int i = 0; i < count; i++){
        assign_input(positions[offset + i], inputs[i], targets[i]);
    }

    return count;
}
}    // namespace dense

namespace nmm {
inline void assign_input(Position& p, Input& input, Data& output) {
    PositionIterator it {p};
    output.clear();
    output(it.score) = 1;
    input.indices.clear();
    while (it.hasNext()) {
        it.next();
        int offset = 64 * 6;
        if (it.piece / 6 == it.activePlayer) {
            offset = 0;
        }
        input.indices.push_back(offset + (it.piece % 6) * 64 + it.sq);
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
}    // namespace nmm

#endif    // DIFFERENTIATION_MAPPINGS_H
