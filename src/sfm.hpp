#pragma once

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

// std
#include <string>
#include <vector>

struct Problem {
    int ref_image_id;
    std::vector<int> src_image_ids;
};

struct Camera {
  float K[9];
  float R[9];
  float t[3];
  int height;
  int width;
  float depth_min;
  float depth_max;
};

class SfM {
public:
  std::vector<cv::Mat> images;
  std::vector<Camera> cameras;
  void InputInitialization(const std::string &dense_folder,
                           const Problem &problem);

private:
};
