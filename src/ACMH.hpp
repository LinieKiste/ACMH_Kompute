#pragma once
#include "sfm.hpp"

#include "kompute/Core.hpp"
#include "kompute/Kompute.hpp"
#include <sail-c++/sail-c++.h>
#include <glm/vec2.hpp>

// std
#include <memory>
#include <cstdint>
#include <cstring>

class ACMH {
public:
  ACMH(std::string img_folder);
  void RunPatchMatch();

private:
  SfM sfm;
  std::vector<float> pixels;
  uint nr_images;
};
