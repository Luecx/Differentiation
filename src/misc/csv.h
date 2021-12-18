
//
// Created by Luecx on 18.12.2021.
//

#ifndef DIFFERENTIATION_SRC_MISC_CSV_H_
#define DIFFERENTIATION_SRC_MISC_CSV_H_

#include <cstdarg>
#include <fstream>
struct CSVWriter {

    std::ofstream csv_file{};

    CSVWriter(std::string res){
        csv_file = std::ofstream {res};
    }

    virtual ~CSVWriter() {
        csv_file.close();
    }

    template<typename... Args>
    void write(Args... args){

        ([&] (auto & input)
        {
          csv_file << input << ",";
        } (args), ...);

        csv_file <<"\n" << std::flush;
    }

};

#endif    // DIFFERENTIATION_SRC_MISC_CSV_H_
