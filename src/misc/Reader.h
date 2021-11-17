//
// Created by Luecx on 23.02.2021.
//

#ifndef DIFFERENTIATION_READER_H
#define DIFFERENTIATION_READER_H

#include <bitset>
#include <ostream>
#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include "logging.h"


typedef int8_t Piece;
typedef int8_t Square;
typedef int8_t PieceType;
typedef int8_t Rank;
typedef int8_t File;
typedef bool Color;

#define mirror(s) squareIndex(7 - rankIndex(s), fileIndex(s))

inline Rank rankIndex(Square square_index) { return square_index >> 3; }

inline File fileIndex(Square square_index) { return square_index & 7; }

inline Square squareIndex(Rank rank, File file) { return 8 * rank + file; }

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

struct BitEntry{
    int index;
    bool occupied;
    Color color;
    Piece piece;
    int bits;

    friend std::ostream &operator<<(std::ostream &os, const BitEntry &entry);
};

extern BitEntry lookUpTable[64];
extern BitEntry lookUpTablePieces[12];

enum ScoreFormat{
    CP,
    CP_SEER,
    P,
    WDL,
    UCI_MOVE,
    AGE
};

inline void initLookUpTable(){

    for(uint8_t i = 0; i < 64; i++){

        lookUpTable[i].index = i;
        lookUpTable[i].bits  = 1;

        // check if its occupied
        if(i & (1 << 0)){
            lookUpTable[i].occupied = true;
        }else{
            continue;
        }

        // check which color is on it
        if(i & (1 << 1)){
            lookUpTable[i].color = BLACK;
        }else{
            lookUpTable[i].color = WHITE;
        }


        // check if its a pawn
        if(!(i & (1 << 2))) {
            lookUpTable[i].piece = PAWN;
            lookUpTable[i].bits = 3;
            continue;
        }
        // check if its a knight, bishop or rook
        switch((i >> 3) & 3){
            case 0:{
                lookUpTable[i].piece = KNIGHT;
                lookUpTable[i].bits = 5;
                continue;
            }
            case 1:{
                lookUpTable[i].piece = BISHOP;
                lookUpTable[i].bits = 5;
                continue;
            }
            case 2:{
                lookUpTable[i].piece = ROOK;
                lookUpTable[i].bits = 5;
                continue;
            }
        }

        // has to be a king or queen
        if ((i & (1 << 5))) {
            lookUpTable[i].piece = KING;
            lookUpTable[i].bits = 6;
        }else{
            lookUpTable[i].piece = QUEEN;
            lookUpTable[i].bits = 6;
        }



    }

    for(uint8_t i = 64; i >= 0; i--){
        if(lookUpTable[i].occupied){
            lookUpTablePieces[6 * lookUpTable[i].color + lookUpTable[i].piece] = lookUpTable[i];
        }
        if(i == 0){
            break;
        }
    }


}

struct Position{

    std::bitset<8*24> bits{};

    bool set(const std::string &fen, ScoreFormat scoreFormat = CP) {
        // set the active player to white if it is not specified later on
        Color activePlayer = WHITE;
        // temporarely store the pieces in a table with 64 entries and initialise those pieces to -1 (no piece)
        Piece pieces[64]{};
        for(signed char & piece : pieces) piece = -1;
        // keep track of the row/col
        int row = 7;
        int col = 0;
        for(char c:fen){

            // break if enough squares have been read
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
                col ++;
            }
        }
        // parse the side to move
        // find the first space. if there is one as well as a char after that, parse that char
        if(fen.find(' ') < fen.length() - 1){
            int pos = fen.find_first_of(' ');
            char stm = fen[pos+1];
            if(stm == 'w'){
                activePlayer = WHITE;
            }else{
                activePlayer = BLACK;
            }
        }

        // set the active player bit in the bitset
        bits.set(0,activePlayer);

        // set the remaining bits for the position
        int bitIndex = 1;
        for(int r = 0; r < 8; r ++){
            for(int f = 0; f < 8; f++){
                int id = r * 8 + f;
                if(pieces[id] < 0){
                    bitIndex ++;
                    continue;
                }
                BitEntry b = lookUpTablePieces[pieces[id]];
                std::bitset<8*24> h(b.index);
                bits |= (h << bitIndex);
                bitIndex += b.bits;
            }
        }


        int16_t score = 0;
        // parse a score if one has been specified.
        // check if the format equals age format which includes wdl and score [WDL] score
        if(scoreFormat == AGE && fen.find('[') < fen.length()-1){
            auto posWDLStart = fen.find_first_of('[');
            auto posWDLEnd   = fen.find_first_of(']');
            std::string wdl = fen.substr(posWDLStart+1,posWDLEnd - posWDLStart - 1);
            std::string scr = fen.substr(posWDLEnd  +1);

            int    wdlValue = (int)round(std::stod(wdl)*2);
            int    scrValue = std::stoi(scr);

            int16_t finalScore = scrValue;

            if(wdlValue == 1){
                // pass
            }else if(wdlValue == 2){
                // white won
                finalScore += 20000; // > 10000
            }else if(wdlValue == 0){
                // black won
                finalScore -= 20000; // < -10000
            }

            score = finalScore;
        }
        // parse singular scores or wdl values
        // this is the case if a semicolon (;) has been found. the value after the semicolon (;) is considered to
        // be the score. it will be stored in the last 16 bits
        else if(fen.find(';') < fen.length() - 1 && fen.find('#') >= fen.length()){

            auto pos = fen.find_first_of(';');
            std::string relevant = fen.substr(pos+1, fen.size());

            if(scoreFormat == CP){
                score = stoi(relevant);
//                if(abs(score) > 1500)
//                    return false;
            }

            if(scoreFormat == P){
                score = round(stod(relevant) * 100);
//                if(abs(score) > 1500)
//                    return false;
            }

            if(scoreFormat == CP_SEER){
                score = round(stod(relevant) * 400 / 1024);
            }

            if(scoreFormat == WDL){
                score = round(stod(relevant) * 2000 - 1000);
            }

            if(scoreFormat == UCI_MOVE){
                File f1 = fen.at(pos+1) - 'a';
                File f2 = fen.at(pos+3) - 'a';
                Rank r1 = fen.at(pos+2) - '1';
                Rank r2 = fen.at(pos+4) - '1';
                Square s1 = f1 + r1 * 8;
                Square s2 = f2 + r2 * 8;
                score = s1 * 64 + s2;
                if (score >= 64 * 64){
                    std::cout << fen << std::endl;
                    exit(-1);
                }
            }
        }else{
            return false;
        }

        // set the score bits in the bitset. Since we cannot store unsigned values, we need to add 1 << 16 first
        bits |= (std::bitset<8*24>(score + (1 << 16)) << (22 * 8));

        return true;
    }
};

struct PositionIterator{

private:
    // the last index of the last entry we parsed
    int read_index = 0;
    // the position to parse from
    Position p;
public:

    // the active player for the position. does not change with "next()"
    Color      activePlayer;
    // the last piece which has been parsed
    Piece      piece{};
    // the last square which has been parsed
    Square     sq = -1;
    // the last 16 bits are used for a score which might have been read from the fen
    int16_t    score = 0;

    explicit PositionIterator(Position& p){
        // extract the active player which is stored in the very first bit
        activePlayer = p.bits.test(0);
        // when reading the score, we need to consider to move it from being unsigned to signed
        score        = (p.bits >> (8*22)).to_ulong() - (1 << 16);
        this->p = p;
    };

    bool hasNext() const{
        // check if there is a 1 left in the bitset which would indicate a piece
        return p.bits._Find_next(read_index) < 22 * 8;
    }

    void next(){
        // get the next active bit index
        int n = p.bits._Find_next(read_index);
        // we compute the square this bit is on depending on how many bits we have skipped.
        // since read_index points to the end of the previous section, n - read_index is 1 if no bits have been skipped
        sq += (n - read_index);
        // figure out the BitEntry we are looking at by extracting the 6 relevant bits
        BitEntry en = lookUpTable[((p.bits >> n) & std::bitset<192>{63}).to_ulong()];
        // compute the piece
        piece = en.piece + en.color * 6;
        // compute the next index to read from
        read_index = n + en.bits - 1;
    }

};

inline void read_positions_txt(const std::string &file, std::vector<Position> *positions, ScoreFormat scoreFormat = CP, int max_lines=-1, int reserve=0) {
    std::ifstream infile(file);
    std::string line;

    if(!infile.is_open()){
        std::cout << "could not open: " << file << std::endl;
        logging::write("could not open " + file);
    }

    if(reserve != 0){
        positions->reserve(reserve);
        std::cout << "increased capacity to fit " << positions->capacity() << " entries" << std::endl;
    }

    int count = 0;
    while (std::getline(infile, line)){
        Position p{};
        if(p.set(line, scoreFormat)){
            if(positions->size() % 1000 == 0){
                printf("\r[Loading positions] Current size=%d", positions->size());
                fflush(stdout);
            }
            positions->push_back(p);
            count ++;
            if(count == max_lines){
                break;
            }
        }else{
            std::cout << "could not load: " << line << std::endl;
        }
    }
    std::cout << std::endl;
    logging::write("loaded " + std::to_string(count) + " positions from: " + file);
}
inline void read_positions_bin(const std::string &file, std::vector<Position> *positions, int max_chunks=-1) {

    // read amount of values contained
    uint64_t        num;
    FILE *f = fopen(file.c_str(), "rb");
    if(f == nullptr){
        logging::write("could not open " + file);
        return;
    }
    fread(&num             , sizeof(uint64_t), 1  , f);

    // compute chunks
    int chunks = std::ceil(num / 1e6);

    // correct chunks
    if(max_chunks >= 0 && max_chunks < chunks){
        chunks = max_chunks;
        num    = chunks * 1e6;
    }

    int offset = positions->size();

    positions->resize(positions->size() + num);

    for(int c = 0; c < chunks; c++){
        int start = c * 1e6 + offset;
        int end   = c * 1e6 + 1e6 + offset;
        if(end > positions->size()) end = positions->size();
        fread(&positions->at(start), sizeof(Position), end-start, f);
        printf("\r[Reading positions] Current count=%d", end);
        fflush(stdout);
    }
    std::cout << std::endl;
    logging::write("loaded " + std::to_string(num) + " positions from: " + file);

    fclose(f);
}
inline void write_positions_bin(const std::string &file, std::vector<Position> *positions){

    uint64_t        num = positions->size();
    FILE *f = fopen(file.c_str(), "wb");
    fwrite(&num             , sizeof(uint64_t), 1  , f);

    int chunks = std::ceil(positions->size() / 1e6);

    for(int c = 0; c < chunks; c++){
        int start = c * 1e6;
        int end   = c * 1e6 + 1e6;
        if(end > positions->size()) end = positions->size();
        fwrite(&positions->at(start), sizeof(Position), end-start, f);
        printf("\r[Writing positions] Current count=%d", end);
    }
    std::cout << std::endl;
    logging::write("written " + std::to_string(num) + " positions to: " + file);

    fclose(f);

}


#endif //DIFFERENTIATION_READER_H
