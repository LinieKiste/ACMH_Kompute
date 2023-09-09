#pragma once

#include <vector>
#include <cstdint>

struct Shaders{
    std::vector<uint32_t> random_init;
    std::vector<uint32_t> black_pixel_update;
    std::vector<uint32_t> red_pixel_update;
    std::vector<uint32_t> get_depth_and_normal;
    std::vector<uint32_t> black_filter;
    std::vector<uint32_t> red_filter;
};

class Cache {
public:
  Cache();

  Shaders shaders;
private:
};
