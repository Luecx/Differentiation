//
// Created by Luecx on 03.03.2021.
//

#ifndef DIFFERENTIATION_MAPPINGS_H
#define DIFFERENTIATION_MAPPINGS_H


#include "Reader.h"
#include "Data.h"

namespace nnue {
    int index(Color active, Color kingColor, Color pieceColor, Square ksq, Square psq, PieceType p);

    int index_factor_1(Color active, Color kingColor, Color pieceColor, Square ksq, Square psq, PieceType p);

    void assign_input(Position &p, Input &input, Data& output, bool original=false);

    int assign_inputs_batch(std::vector<Position> &positions, int offset, std::vector<Input> &inputs, std::vector<Data> &targets);

    void reduce_matrix(Data& original, Data& reduced);
};


namespace ataxx{

    struct Sample {
        Sample() : activePieces{}, passivePieces{}, target{} {
        }

        void set(const std::string& fen){

            std::string word;

            std::stringstream ss{fen};
            std::uint64_t mask_x = 0ULL;
            std::uint64_t mask_o = 0ULL;

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
                for (const auto &c : word) {
                    switch (c) {
                        case 'x':
                            mask_x ^= (1ULL << idx);
                            idx++;
                            break;
                        case 'o':
                            mask_o ^= (1ULL << idx);
                            idx++;
                            break;
                        case '-':
                            idx++;
                            break;
                        case '1':
                        case '2':
                        case '3':
                        case '4':
                        case '5':
                        case '6':
                        case '7':
                            idx += c - '0';
                            break;
                        case '/':
                            break;
                        default:
                            break;
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
        float target;
    };

    void load_positions(const std::string& fil, std::vector<Sample> &samples);

    void assign_input(Sample &p, Input &input, Data& output);

    int assign_inputs_batch(std::vector<Sample> &positions, int offset, std::vector<Input> &inputs, std::vector<Data> &targets);



}


namespace dense{
    int index(Square psq, Piece p);

//    int index_factor_1(Color active, Color kingColor, Color pieceColor, Square ksq, Square psq, PieceType p);

    void assign_input(Position &p, Input &input, Data& output);

    int assign_inputs_batch(std::vector<Position> &positions, int offset, std::vector<Input> &inputs, std::vector<Data> &targets);
}

#endif //DIFFERENTIATION_MAPPINGS_H
