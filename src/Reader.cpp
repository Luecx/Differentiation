//
// Created by Luecx on 23.02.2021.
//

#include "Reader.h"

std::ostream &operator<<(std::ostream &os, const BitEntry &entry) {
    os << "occupied: " << entry.occupied << " color: " << entry.color << " piece: " << (int)entry.piece << " bits: "
       << entry.bits;
    return os;
}
