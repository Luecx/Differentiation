/****************************************************************************************************
 *                                                                                                  *
 *                                                FFES                                              *
 *                                          by. Finn Eggers                                         *
 *                                                                                                  *
 *                    FFES is free software: you can redistribute it and/or modify                  *
 *                it under the terms of the GNU General Public License as published by              *
 *                 the Free Software Foundation, either version 3 of the License, or                *
 *                                (at your option) any later version.                               *
 *                       FFES is distributed in the hope that it will be useful,                    *
 *                   but WITHOUT ANY WARRANTY; without even the implied warranty of                 *
 *                   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the                  *
 *                            GNU General Public License for more details.                          *
 *                 You should have received a copy of the GNU General Public License                *
 *                   along with FFES.  If not, see <http://www.gnu.org/licenses/>.                  *
 *                                                                                                  *
 ****************************************************************************************************/

//
// Created by Luecx on 16.12.2021.
//

#ifndef DIFFERENTIATION_SRC_MISC_TIMER_H_
#define DIFFERENTIATION_SRC_MISC_TIMER_H_

#include <chrono>
template<class TimeT = std::chrono::milliseconds, class ClockT = std::chrono::steady_clock> class Timer {
    using timep_t  = typename ClockT::time_point;
    timep_t _start = ClockT::now(), _end = {};

    public:
    void tick() {
        _end   = timep_t {};
        _start = ClockT::now();
    }

    void tock() { _end = ClockT::now(); }

    template<class TT = TimeT> TT duration() const {
//        gsl_Expects(_end != timep_t {} && "toc before reporting");
        return std::chrono::duration_cast<TT>(_end - _start);
    }
};

#endif    // DIFFERENTIATION_SRC_MISC_TIMER_H_
