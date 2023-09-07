#pragma once
#include "sfm.hpp"

#include "kompute/Core.hpp"
#include "kompute/Kompute.hpp"
#include <glm/glm.hpp>

// std
#include <filesystem>
#include <iostream>
#include <string>
#include <memory>
#include <cstdint>
#include <cstring>

struct Parameters {
  int max_iterations = 3;
  int patch_size = 11;
  int num_images = 5;
  int radius_increment = 2;
  float sigma_spatial = 5.0f;
  float sigma_color = 3.0f;
  int top_k = 4;
  float depth_min = 0.0f;
  float depth_max = 1.0f;
  uint geom_consistency = false; // uint does not work because of padding
};

struct PushConstants {
  Parameters params;
  int iter;
  float timestamp;
};

class ACMH {
public:
  ACMH(std::string dense_folder, int ref_image_id,
       std::vector<int> src_image_ids);
  void init_depths(std::string dense_folder, int ref_image_id,
                   std::vector<int> src_image_ids);
  void VulkanSpaceInitialization(const std::string &dense_folder,
                                 int ref_image_id,
                                 std::vector<int> src_image_ids);
  void RunPatchMatch();
  void SetGeomConsistencyParams();
  int GetReferenceImageWidth();
  int GetReferenceImageHeight();
  glm::vec4 GetPlaneHypothesis(const int index);
  void SetPlaneHypothesis(const int index, glm::vec4 hyp);
  float GetCost(const int index);

  kp::Manager mgr;
  std::vector<float> plane_hypotheses_host; // vector of glm::vec4
  std::vector<float> costs_host;

private:
  SfM sfm;
  std::vector<float> pixels;
  std::vector<cv::Mat> depths;

  std::vector<std::shared_ptr<kp::Tensor>> kp_params;
  struct {
    std::shared_ptr<kp::TensorT<float>> image_tensor;
    std::shared_ptr<kp::TensorT<float>> plane_hypotheses_tensor;
    std::shared_ptr<kp::TensorT<float>> costs_tensor;
    std::shared_ptr<kp::TensorT<float>> camera_tensor;
    std::shared_ptr<kp::TensorT<float>> random_states_tensor;
    std::shared_ptr<kp::TensorT<uint>> selected_views_tensor;
    std::shared_ptr<kp::TensorT<float>> depths_tensor;
    } tensors;
  struct {
    std::vector<uint32_t> random_init;
    std::vector<uint32_t> black_pixel_update;
    std::vector<uint32_t> red_pixel_update;
    std::vector<uint32_t> get_depth_and_normal;
    std::vector<uint32_t> black_filter;
    std::vector<uint32_t> red_filter;
  } shaders;

    Parameters params;
  };
