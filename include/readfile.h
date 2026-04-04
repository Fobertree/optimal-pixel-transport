//
// Created by Alexander Liu on 3/4/26.
//

#ifndef OPTIMALPIXELTRANSPORT_READFILE_H
#define OPTIMALPIXELTRANSPORT_READFILE_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>

const std::string PREFIX = "Shaders/";

void list_all_paths();

std::string read_wgsl_file(const std::string &filepath) {
    std::string full_path = PREFIX + filepath;
    // I don't think this supports raw strings
    try {
        std::ifstream file(full_path);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open the file: " << full_path << std::endl;
            list_all_paths();
            std::cout << "FAILURE";
            return "";
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }
    catch (std::filesystem::filesystem_error &error) {
        std::cerr << full_path << " is invalid" << std::endl;
        std::cout << "FAILURE" << std::endl;
    }
    return "";
}

void list_all_paths() {
    for (auto &p: std::filesystem::recursive_directory_iterator("/")) {
        std::cout << p.path() << std::endl;
    }
}

#endif //OPTIMALPIXELTRANSPORT_READFILE_H
