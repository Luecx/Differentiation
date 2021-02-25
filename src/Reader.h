//
// Created by Luecx on 23.02.2021.
//

#ifndef DIFFERENTIATION_READER_H
#define DIFFERENTIATION_READER_H

#include <bitset>


typedef int8_t Piece;
typedef int8_t Square;
typedef int8_t PieceType;
typedef bool Color;

enum PieceTypes{
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING
};

enum Colors{
    WHITE,
    BLACK,
};

enum Pieces{
    WHITE_PAWN,
    WHITE_KNIGHT,
    WHITE_BISHOP,
    WHITE_ROOK,
    WHITE_QUEEN,
    WHITE_KING,
    BLACK_PAWN,
    BLACK_KNIGHT,
    BLACK_BISHOP,
    BLACK_ROOK,
    BLACK_QUEEN,
    BLACK_KING
};



struct Position{

    std::bitset<8*24> bits{};

    void set(const std::string &fen) {
        int index = 0;
        Piece pieces[64]{};
        for(int i = 0; i < 64; i++) pieces[i] = -1;
        int row = 7;
        int col = 0;
        for(char c:fen){

            if(row < 0) break;
            if(col >= 8) col = 0;

            if(c == ' ') break;
            if(c == '/'){
                row --;
                col = 0;
                continue;
            }else if(c < '9' && c >= '0'){
                col += (c - '0');
            }else{

                Color     cl = c > 'a' ? BLACK:WHITE;
                c            = c - (cl == BLACK ? 'a'-'A':0);

                switch(c){
                    case 'P': pieces[col + row * 8] = cl * 6 + PAWN    ; break;
                    case 'N': pieces[col + row * 8] = cl * 6 + KNIGHT  ; break;
                    case 'B': pieces[col + row * 8] = cl * 6 + BISHOP  ; break;
                    case 'R': pieces[col + row * 8] = cl * 6 + ROOK    ; break;
                    case 'Q': pieces[col + row * 8] = cl * 6 + QUEEN   ; break;
                    case 'K': pieces[col + row * 8] = cl * 6 + KING    ; break;
                }
//                std::cout << col + row * 8 << "  " << (int) pieces[col + row * 8] << std::endl;
//                std::cout << col << "  v  " << row << std::endl;
                col ++;
            }
        }

        int bitIndex = 0;
        for(int r = 0; r < 8; r ++){
            for(int f = 0; f < 8; f++){
                int id = r * 8 + f;
                if(pieces[id] >= 0){
                    bits.set(bitIndex);
                    bits.set(bitIndex + 1, (pieces[id] / 8) % 2 == 1);
                    bits.set(bitIndex + 2, (pieces[id] / 4) % 2 == 1);
                    bits.set(bitIndex + 3, (pieces[id] / 2) % 2 == 1);
                    bits.set(bitIndex + 4, (pieces[id] / 1) % 2 == 1);
                    bitIndex += 5;
                }else{
                    bitIndex += 1;
                }
            }
        }
    }
};

struct PositionIterator{

    Piece      piece;
    Square     sq = -1;


    int read_index = -1;
    Position p;

    PositionIterator(Position& p){
        this->p = p;
    };

    bool hasNext(){
        return p.bits._Find_next(read_index) != p.bits.size();
    }

    void next(){
        int n = p.bits._Find_next(read_index);
//        std::cerr << read_index << "  " << n << std::endl;
        sq += (n - read_index);

        piece =
              8 * p.bits.test(n + 1)
            + 4 * p.bits.test(n + 2)
            + 2 * p.bits.test(n + 3)
            + 1 * p.bits.test(n + 4);


        read_index = n + 4;
    }



};

inline void read_positions(const std::string &file) {

}


#endif //DIFFERENTIATION_READER_H
