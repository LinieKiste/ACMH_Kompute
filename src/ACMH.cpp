#include "ACMH.hpp"
#include "helpers.hpp"

#define LOAD_RENDERDOC

#ifdef LOAD_RENDERDOC
#include "renderdoc_app.h"
#endif

namespace fs = std::filesystem;

ACMH::ACMH(std::string dense_folder, int ref_image_id,
           std::vector<int> src_image_ids)
    : sfm(dense_folder, ref_image_id, src_image_ids) {
  params.num_images = sfm.params.num_images;
  params.depth_min = sfm.params.depth_min;
  params.depth_max = sfm.params.depth_max;

  // load shaders
  shaders.random_init = helpers::loadShader("random_init.comp");
  shaders.black_pixel_update = helpers::loadShader("black_pixel_update.comp");
  shaders.red_pixel_update = helpers::loadShader("red_pixel_update.comp");
  shaders.get_depth_and_normal = helpers::loadShader("get_depth_normal.comp");
  shaders.black_filter = helpers::loadShader("black_filter.comp");
  shaders.red_filter = helpers::loadShader("red_filter.comp");
}

void ACMH::init_depths(std::string dense_folder, int ref_image_id,
                 std::vector<int> src_image_ids) {
  depths.clear();

  std::stringstream result_path;
  result_path << dense_folder << "/ACMH"
              << "/2333_" << std::setw(8) << std::setfill('0')
              << ref_image_id;
  std::string result_folder = result_path.str();
  std::string depth_path = result_folder + "/depths.dmb";
  cv::Mat_<float> ref_depth;
  helpers::readDepthDmb(depth_path, ref_depth);
  depths.push_back(ref_depth);

  size_t num_src_images = src_image_ids.size();
  for (size_t i = 0; i < num_src_images; ++i) {
    std::stringstream result_path;
    result_path << dense_folder << "/ACMH"
                << "/2333_" << std::setw(8) << std::setfill('0')
                << src_image_ids[i];
    std::string result_folder = result_path.str();
    std::string depth_path = result_folder + "/depths.dmb";
    cv::Mat_<float> depth;
    helpers::readDepthDmb(depth_path, depth);
    depths.push_back(depth);
  }
}

std::vector<float> mat_to_vec(cv::Mat &image) {
    return image.reshape(1, image.total()*image.channels());
}

// not sure about alignment, might be possible with single copy
std::vector<float> cams_to_vec(std::vector<Camera> cameras) {
  std::vector<float> vec;
  for (auto camera : cameras) {
    vec.insert(vec.end(), camera.K.cbegin(), camera.K.cend());
    vec.insert(vec.end(), camera.R.cbegin(), camera.R.cend());
    vec.insert(vec.end(), camera.t.cbegin(), camera.t.cend());
    vec.push_back(static_cast<float>(camera.height));
    vec.push_back(static_cast<float>(camera.width));
    vec.push_back(camera.depth_min);
    vec.push_back(camera.depth_max);
  }
  return vec;
}

void ACMH::VulkanSpaceInitialization(const std::string &dense_folder,
                                     int ref_image_id,
                                     std::vector<int> src_image_ids) {
  int num_images = (int)sfm.images.size();

  // image tensor
  std::vector<float> img_tensor_data;
  int ref_no_pixels = sfm.cameras[0].height * sfm.cameras[0].width;
  for (int i = 0; i < num_images; ++i) {
    auto image_vec = mat_to_vec(sfm.images[i]);
    img_tensor_data.insert(img_tensor_data.end(), image_vec.begin(),
                           image_vec.end());

  }

  plane_hypotheses_host =
      std::vector<float>(4 * ref_no_pixels); // glm::vec4 per pixel
  costs_host = std::vector<float>(ref_no_pixels);
  std::vector<float> camera_data_host = cams_to_vec(sfm.cameras);
  auto random_states_host = std::vector<float>(ref_no_pixels);
  auto selected_views_host = std::vector<uint32_t>(ref_no_pixels);

  std::vector<float> depth_tensor_data;
  if (params.geom_consistency) {
    for (int i = 0; i < num_images; ++i) {
      auto depth_vec = mat_to_vec(depths[i]);
      depth_tensor_data.insert(depth_tensor_data.end(), depth_vec.begin(),
                               depth_vec.end());
    }

    std::stringstream result_path;
    result_path << dense_folder << "/ACMH"
                << "/2333_" << std::setw(8) << std::setfill('0')
                << ref_image_id;
    std::string result_folder = result_path.str();
    std::string depth_path = result_folder + "/depths.dmb";
    std::string normal_path = result_folder + "/normals.dmb";
    std::string cost_path = result_folder + "/costs.dmb";
    cv::Mat_<float> ref_depth;
    cv::Mat_<cv::Vec3f> ref_normal;
    cv::Mat_<float> ref_cost;
    helpers::readDepthDmb(depth_path, ref_depth);
    // Why?
    depths.push_back(ref_depth);
    helpers::readNormalDmb(normal_path, ref_normal);
    helpers::readDepthDmb(cost_path, ref_cost);
    int width = ref_depth.cols;
    int height = ref_depth.rows;
    for (int col = 0; col < width; ++col) {
      for (int row = 0; row < height; ++row) {
        int center = row * width + col;
        glm::vec4 plane_hypothesis;
        plane_hypothesis.x = ref_normal(row, col)[0];
        plane_hypothesis.y = ref_normal(row, col)[1];
        plane_hypothesis.z = ref_normal(row, col)[2];
        plane_hypothesis.w = ref_depth(row, col);
        SetPlaneHypothesis(center, plane_hypothesis);
        costs_host[center] = ref_cost(row, col);
      }
    }
  } else {
    depth_tensor_data.resize(ref_no_pixels);
  }

  // image tensor
  tensors.image_tensor = mgr.tensor(img_tensor_data);
  // plane hypotheses tensor
  tensors.plane_hypotheses_tensor = mgr.tensor(plane_hypotheses_host);
  // costs tensor
  tensors.costs_tensor = mgr.tensor(costs_host);
  // camera tensor
  tensors.camera_tensor = mgr.tensor(camera_data_host);
  // random states tensor
  tensors.random_states_tensor = mgr.tensorT(random_states_host);
  // selected views tensor
  tensors.selected_views_tensor = mgr.tensorT(selected_views_host);
  // depths tensor
  tensors.depths_tensor = mgr.tensorT(depth_tensor_data);

  kp_params = {tensors.image_tensor,         tensors.plane_hypotheses_tensor,
               tensors.costs_tensor,         tensors.camera_tensor,
               tensors.random_states_tensor, tensors.selected_views_tensor,
               tensors.depths_tensor};
}

void ACMH::RunPatchMatch() {
  const int width = sfm.cameras[0].width;
  const int height = sfm.cameras[0].height;

  int BLOCK_W = 32;
  int BLOCK_H = (BLOCK_W / 2);

  glm::uvec3 grid_size_randinit;
  grid_size_randinit.x = (width + 16 - 1) / 16;
  grid_size_randinit.y = (height + 16 - 1) / 16;
  grid_size_randinit.z = 1;
  glm::uvec3 grid_size_checkerboard;
  grid_size_checkerboard.x = (width + BLOCK_W - 1) / BLOCK_W;
  grid_size_checkerboard.y = ((height / 2) + BLOCK_H - 1) / BLOCK_H;
  grid_size_checkerboard.z = 1;

  int max_iterations = 3;

  auto algo_random_init =
      mgr.algorithm(kp_params, shaders.random_init,
                    kp::Workgroup({grid_size_randinit.x, grid_size_randinit.y,
                                   grid_size_randinit.z}),
                                   {}, std::vector<PushConstants>{{params, 0}});

#ifdef LOAD_RENDERDOC
  // INIT RENDERDOC
  RENDERDOC_API_1_1_2 *rdoc_api = NULL;
  if (void *mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD)) {
    pRENDERDOC_GetAPI RENDERDOC_GetAPI =
        (pRENDERDOC_GetAPI)dlsym(mod, "RENDERDOC_GetAPI");
    int ret =
        RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_1_2, (void **)&rdoc_api);
    assert(ret == 1);
  }
#endif

  // RANDOM INITIALIZATION
  float seed = 0.75; // magic number
  auto sequence = mgr.sequence();
  sequence->record<kp::OpTensorSyncDevice>(kp_params)
      ->record<kp::OpAlgoDispatch>(
          algo_random_init, std::vector<PushConstants>{{params, 0, seed}})
      ->eval();

  auto algo_black_pixel_update =
      mgr.algorithm(kp_params, shaders.black_pixel_update,
                    kp::Workgroup({grid_size_checkerboard.x, grid_size_checkerboard.y,
                                   grid_size_checkerboard.z}),
                                   {}, std::vector<PushConstants>{{params, 0}});
  auto algo_red_pixel_update =
      mgr.algorithm(kp_params, shaders.red_pixel_update,
                    kp::Workgroup({grid_size_checkerboard.x, grid_size_checkerboard.y,
                                   grid_size_checkerboard.z}),
                                   {}, std::vector<PushConstants>{{params, 0}});

  // TODO: use push constant to distinguish red and black
  for (int i = 0; i < max_iterations; ++i) {
#ifdef LOAD_RENDERDOC
  if (rdoc_api && params.geom_consistency)
    rdoc_api->StartFrameCapture(NULL, NULL);
#endif
    mgr.sequence()
        ->record<kp::OpAlgoDispatch>(algo_black_pixel_update,
                                     std::vector<PushConstants>({{params, i}}))
        ->eval();

#ifdef LOAD_RENDERDOC
  if (rdoc_api && params.geom_consistency)
    rdoc_api->EndFrameCapture(NULL, NULL);
#endif

    mgr.sequence()
        ->record<kp::OpAlgoDispatch>(algo_red_pixel_update,
                                     std::vector<PushConstants>({{params, i}}))
        ->eval();
    printf("iteration: %d\n", i);
  }

  auto algo_depth_normal =
      mgr.algorithm(kp_params, shaders.get_depth_and_normal,
                    kp::Workgroup({grid_size_randinit.x, grid_size_randinit.y,
                                   grid_size_randinit.z}),
                                   {}, std::vector<PushConstants>{{params, 0}});
  mgr.sequence()
        ->record<kp::OpAlgoDispatch>(algo_depth_normal,
                                     std::vector<PushConstants>({{params, 0}}))
        ->eval();

  auto algo_black_pixel_filter =
      mgr.algorithm(kp_params, shaders.black_filter,
                    kp::Workgroup({grid_size_checkerboard.x, grid_size_checkerboard.y,
                                   grid_size_checkerboard.z}),
                                   {}, std::vector<PushConstants>{{params, 0}});
  auto algo_red_pixel_filter =
      mgr.algorithm(kp_params, shaders.red_filter,
                    kp::Workgroup({grid_size_checkerboard.x, grid_size_checkerboard.y,
                                   grid_size_checkerboard.z}),
                                   {}, std::vector<PushConstants>{{params, 0}});

  // TODO: use push constants to distinguish red and black
  mgr.sequence()
      ->record<kp::OpAlgoDispatch>(algo_black_pixel_filter,
                                   std::vector<PushConstants>({{params, 0}}))
      ->eval()
      ->record<kp::OpAlgoDispatch>(algo_red_pixel_filter,
                                   std::vector<PushConstants>({{params, 0}}))
      ->eval();
       
  mgr.sequence()->record<kp::OpTensorSyncLocal>(kp_params)->eval();
  this->plane_hypotheses_host = tensors.plane_hypotheses_tensor->vector();
  this->costs_host = tensors.costs_tensor->vector();
}

// helpers
void ACMH::SetGeomConsistencyParams() {
  params.geom_consistency = true;
  params.max_iterations = 2;
}

int ACMH::GetReferenceImageWidth() { return sfm.cameras[0].width; }

int ACMH::GetReferenceImageHeight() { return sfm.cameras[0].height; }

// TODO: 80% chance this is wrong
glm::vec4 ACMH::GetPlaneHypothesis(const int index) {
  return *reinterpret_cast<glm::vec4 *>(plane_hypotheses_host.data() + (index * 4));
}
void ACMH::SetPlaneHypothesis(const int index, glm::vec4 hyp) {
  glm::vec4* ptr = reinterpret_cast<glm::vec4 *>(plane_hypotheses_host.data() + (index * 4));
  *ptr = hyp;
}

float ACMH::GetCost(const int index) { return costs_host[index]; }
