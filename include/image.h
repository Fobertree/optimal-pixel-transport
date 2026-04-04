//
// Created by Alexander Liu on 3/15/26.
//

#ifndef OPTIMALPIXELTRANSPORT_IMAGE_H
#define OPTIMALPIXELTRANSPORT_IMAGE_H

#include <stb_image.h>
#include <string>
#include <iostream>
#include <array>
#include <cmath>

// TODO: template Image class to double vs. float
class Image {
public:
    explicit Image(const std::string &image_path) {
        stbi_set_flip_vertically_on_load(1);
        
        data_ = stbi_load(
                image_path.c_str(),
                &width_,
                &height_,
                &channels_,
                0
        );

        if (!data_) {
            std::cout << "Failed: " << stbi_failure_reason() << std::endl;
        }
    }

    ~Image() {
        // char* not covered by RAII
        stbi_image_free(data_);
    }

    std::array<float, 3> get_pixel(int i, int j) {
        if (i < 0 || i >= height_ || j < 0 || j >= width_) return {};
        int index = (i * width_ + j) * channels_;
        // Loads as uint8: 0 - 255
        float R = static_cast<float>(data_[index]) * divisor_;
        float G = static_cast<float>(data_[index + 1]) * divisor_;
        float B = static_cast<float>(data_[index + 2]) * divisor_;

        return {R, G, B};
    }

//    template<std::floating_point T>
    std::array<float, 3> get_pixel(std::array<int, 2> pos) {
        int i = pos[0], j = pos[1];

        if (i < 0 || i >= height_ || j < 0 || j >= width_) return {};
        int index = (i * width_ + j) * channels_;
        // Loads as uint8: 0 - 255
        float R = static_cast<float>(data_[index]) * divisor_;
        float G = static_cast<float>(data_[index + 1]) * divisor_;
        float B = static_cast<float>(data_[index + 2]) * divisor_;

        return {R, G, B};
    }

    std::array<float, 3> rgb_interpolate(float i, float j) {
        // i,j are in [0,1]
        // 2d bilinear interpolation
        // https://math.stackexchange.com/questions/3230376/interpolate-between-4-points-on-a-2d-plane
        float scaled_i = i * static_cast<float>(height_);
        float scaled_j = j * static_cast<float>(width_);
        std::array<float, 3> out_rgb{};

        float y = scaled_i - std::floor(scaled_i);
        float x = scaled_j - std::floor(scaled_j);

        std::array<int, 2> q_00 = {static_cast<int>(scaled_i), static_cast<int>(scaled_j)};
        std::array<int, 2> q_01 = {q_00[0], q_00[1] + 1};
        std::array<int, 2> q_10 = {q_00[0] + 1, q_00[1]};
        std::array<int, 2> q_11 = {q_00[0] + 1, q_00[1] + 1};

        std::array<float, 3> v_00 = get_pixel(q_00),
                v_01 = get_pixel(q_01),
                v_10 = get_pixel(q_10),
                v_11 = get_pixel(q_11);

        out_rgb[0] = (1 - x) * (1 - y) * v_00[0] + x * (1 - y) * v_10[0] + (1 - x) * y * v_01[0] + x * y * v_11[0];
        out_rgb[1] = (1 - x) * (1 - y) * v_00[1] + x * (1 - y) * v_10[1] + (1 - x) * y * v_01[1] + x * y * v_11[1];
        out_rgb[2] = (1 - x) * (1 - y) * v_00[2] + x * (1 - y) * v_10[2] + (1 - x) * y * v_01[2] + x * y * v_11[2];

        return out_rgb;
    }

private:
    static constexpr float divisor_ = 1 / 255.;
    unsigned char *data_;
    int width_ = 0;
    int height_ = 0;
    int channels_ = 0;
};

#endif //OPTIMALPIXELTRANSPORT_IMAGE_H
