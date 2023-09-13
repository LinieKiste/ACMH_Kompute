#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/imgcodecs.hpp>

// std
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

struct Camera {
  std::array<float, 12> K;
  std::array<float, 12> R;
  std::array<float, 4> t;
  float height;
  float width;
  float depth_min;
  float depth_max;
};

class SfM {
public:
  SfM(const std::string &dense_folder, int ref_image_id, std::vector<int> src_image_ids);
  static Camera ReadCamera(const std::string &cam_path);

  struct
  {
    int num_images = 5;
    int max_image_size = 3200;
    float depth_min = 0.0f;
    float depth_max = 1.0f;
    float baseline = 0.54f;
    float disparity_min = 0.0f;
    float disparity_max = 1.0f;
  } params;
  std::vector<cv::Mat> images;
  std::vector<Camera> cameras;

private:
  void get_image_and_camera(int image_id, std::string image_folder,
                            std::string cam_folder);
};
