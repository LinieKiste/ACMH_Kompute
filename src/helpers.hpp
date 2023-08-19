#pragma once
#include <cstdint>
#include <sail-c++/image-c++.h>
#include <vector>
#include <cstring>
#include <fstream>

namespace helpers {
std::vector<uint32_t> compileSource(const std::string &source);
std::vector<float> BPP24_RGB_to_float_rgb(sail::image input);
std::vector<uint8_t> float_rgb_to_BPP24_RGB(void *data, size_t size);
}
