#pragma once
#include "sfm.hpp"

#include "kompute/Core.hpp"
#include "kompute/Kompute.hpp"
#include <glm/glm.hpp>

// std
#include <memory>
#include <cstdint>
#include <cstring>

struct PushConstants {
  Camera camera;
  bool geom_consistency;
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
  float GetCost(const int index);

  kp::Manager mgr;
  glm::vec4 *plane_hypotheses_host;
  std::vector<float> costs_host;

private:
  SfM sfm;
  std::vector<float> pixels;
  std::vector<cv::Mat> depths;

  std::vector<std::shared_ptr<kp::Tensor>> kp_params;

  struct
  {
    int max_iterations = 3;
    int patch_size = 11;
    int num_images = 5;
    int radius_increment = 2;
    float sigma_spatial = 5.0f;
    float sigma_color = 3.0f;
    int top_k = 4;
    bool geom_consistency = false;
  } params;
};
