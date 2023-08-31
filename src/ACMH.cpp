#include "ACMH.hpp"
#include "helpers.hpp"

#include "renderdoc_app.h"

namespace fs = std::filesystem;

ACMH::ACMH(std::string dense_folder, int ref_image_id,
           std::vector<int> src_image_ids)
    : sfm(dense_folder, ref_image_id, src_image_ids) {
  params.depth_min = sfm.params.depth_min;
  params.depth_max = sfm.params.depth_max;
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
  int total_no_pixels = 0;
  for (int i = 0; i < num_images; ++i) {
    auto image_vec = mat_to_vec(sfm.images[i]);
    img_tensor_data.insert(img_tensor_data.end(), image_vec.begin(),
                           image_vec.end());

    total_no_pixels += sfm.cameras[i].height * sfm.cameras[i].width;
  }

    // auto no_pixels = sfm.cameras[0].height * sfm.cameras[0].width;
    // plane_hypotheses_host = std::vector<float>(no_pixels * sizeof(glm::vec4));

    // costs_host = std::vector<float>(no_pixels);
    // auto costs_tensor = mgr.tensor(costs_host);
    // cudaMalloc((void**)&costs_cuda, sizeof(float) * (cameras[0].height * cameras[0].width));

    // TODO: WHAT IS THIS??
    // cudaMalloc((void**)&rand_states_cuda, sizeof(curandState) * (cameras[0].height * cameras[0].width));
    // cudaMalloc((void**)&selected_views_cuda, sizeof(unsigned int) * (cameras[0].height * cameras[0].width));

    // cudaMalloc((void**)&depths_cuda, sizeof(float) * (cameras[0].height * cameras[0].width));

/*
    if (params.geom_consistency) {
        for (int i = 0; i < num_images; ++i) {
            int rows = depths[i].rows;
            int cols = depths[i].cols;

            // cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
            cudaMallocArray(&cuDepthArray[i], &channelDesc, cols, rows);
            cudaMemcpy2DToArray (cuDepthArray[i], 0, 0, depths[i].ptr<float>(), depths[i].step[0], cols*sizeof(float), rows, cudaMemcpyHostToDevice);

            struct cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(cudaResourceDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = cuDepthArray[i];

            struct cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(cudaTextureDesc));
            texDesc.addressMode[0] = cudaAddressModeWrap;
            texDesc.addressMode[1] = cudaAddressModeWrap;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode  = cudaReadModeElementType;
            texDesc.normalizedCoords = 0;

            cudaCreateTextureObject(&(texture_depths_host.images[i]), &resDesc, &texDesc, NULL);
        }
        cudaMalloc((void**)&texture_depths_cuda, sizeof(cudaTextureObjects));
        cudaMemcpy(texture_depths_cuda, &texture_depths_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice);

        std::stringstream result_path;
        result_path << dense_folder << "/ACMH" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
        std::string result_folder = result_path.str();
        std::string depth_path = result_folder + "/depths.dmb";
        std::string normal_path = result_folder + "/normals.dmb";
        std::string cost_path = result_folder + "/costs.dmb";
        cv::Mat_<float> ref_depth;
        cv::Mat_<cv::Vec3f> ref_normal;
        cv::Mat_<float> ref_cost;
        readDepthDmb(depth_path, ref_depth);
        depths.push_back(ref_depth);
        readNormalDmb(normal_path, ref_normal);
        readDepthDmb(cost_path, ref_cost);
        int width = ref_depth.cols;
        int height = ref_depth.rows;
        for (int col = 0; col < width; ++col) {
            for (int row = 0; row < height; ++row) {
                int center = row * width + col;
                float4 plane_hypothesis;
                plane_hypothesis.x = ref_normal(row, col)[0];
                plane_hypothesis.y = ref_normal(row, col)[1];
                plane_hypothesis.z = ref_normal(row, col)[2];
                plane_hypothesis.w = ref_depth(row, col);
                plane_hypotheses_host[center] = plane_hypothesis;
                costs_host[center] = ref_cost(row, col);
            }
        }

        cudaMemcpy(plane_hypotheses_cuda, plane_hypotheses_host, sizeof(float4) * width * height, cudaMemcpyHostToDevice);
        cudaMemcpy(costs_cuda, costs_host, sizeof(float) * width * height, cudaMemcpyHostToDevice);
    }
    */
    // image tensor
    tensors.image_tensor =
        mgr.tensor(img_tensor_data);

    // plane hypotheses tensor
    plane_hypotheses_host = std::vector<float>(4 * total_no_pixels); // glm::vec4 per pixel
    tensors.plane_hypotheses_tensor =
        mgr.tensor(plane_hypotheses_host);

    // costs tensor
    costs_host = std::vector<float>(total_no_pixels);
    tensors.costs_tensor = mgr.tensor(costs_host);

    // camera tensor
    tensors.camera_tensor =
        mgr.tensor(cams_to_vec(sfm.cameras));

    // selected views tensor
    auto selected_views_host = std::vector<uint32_t>(total_no_pixels);
    tensors.selected_views_tensor =
        mgr.tensorT(selected_views_host);

    kp_params = {tensors.image_tensor, tensors.plane_hypotheses_tensor,
                 tensors.costs_tensor, tensors.camera_tensor,
                 tensors.selected_views_tensor};
}

void ACMH::RunPatchMatch() {
  const int width = sfm.cameras[0].width;
  const int height = sfm.cameras[0].height;

  int BLOCK_W = 32;
  int BLOCK_H = (BLOCK_W / 2);

  glm::uvec3 grid_size_randinit;
  grid_size_randinit.x = width;
  grid_size_randinit.y = height;
  grid_size_randinit.z = 1;
  // grid_size_randinit.x = (width + 16 - 1) / 16;
  // grid_size_randinit.y = (height + 16 - 1) / 16;
  // grid_size_randinit.z = 1;
  // glm::uvec3 block_size_randinit;
  // block_size_randinit.x = 16;
  // block_size_randinit.y = 16;
  // block_size_randinit.z = 1;
  /*
      glm::vec3 grid_size_checkerboard;
      grid_size_checkerboard.x = (width + BLOCK_W - 1) / BLOCK_W;
      grid_size_checkerboard.y= ( (height / 2) + BLOCK_H - 1) / BLOCK_H;
      grid_size_checkerboard.z = 1;
      glm::uvec3 block_size_checkerboard;
      block_size_checkerboard.x = BLOCK_W;
      block_size_checkerboard.y = BLOCK_H;
      block_size_checkerboard.z = 1;
  */
  int max_iterations = 3;

  // load shader
  std::ifstream shaderfile("./src/shaders/random_init.comp");
  if (shaderfile.fail())
    throw std::runtime_error("invalid shader path");
  std::stringstream shader;
  shader << shaderfile.rdbuf();

  auto algorithm =
      mgr.algorithm(kp_params, helpers::compileSource(shader.str()),
                    kp::Workgroup({grid_size_randinit.x, grid_size_randinit.y,
                                   grid_size_randinit.z}),
                    {}, std::vector<PushConstants>{{params}});

  // INIT RENDERDOC
  RENDERDOC_API_1_1_2 *rdoc_api = NULL;
  if (void *mod = dlopen("librenderdoc.so", RTLD_NOW | RTLD_NOLOAD)) {
    pRENDERDOC_GetAPI RENDERDOC_GetAPI =
        (pRENDERDOC_GetAPI)dlsym(mod, "RENDERDOC_GetAPI");
    int ret =
        RENDERDOC_GetAPI(eRENDERDOC_API_Version_1_1_2, (void **)&rdoc_api);
    assert(ret == 1);
  }
  if (rdoc_api)
    rdoc_api->StartFrameCapture(NULL, NULL);

  // 4. Run operation synchronously using sequence
  mgr.sequence()
      ->record<kp::OpTensorSyncDevice>(kp_params)
      ->record<kp::OpAlgoDispatch>(
          algorithm,
          std::vector<PushConstants>{{params}}) // Binds push consts
      ->record<kp::OpTensorSyncLocal>(kp_params)
      ->eval();

  if (rdoc_api)
    rdoc_api->EndFrameCapture(NULL, NULL);

  // RandomInitialization<<<grid_size_randinit,
  // block_size_randinit>>>(texture_objects_cuda, cameras_cuda,
  // plane_hypotheses_cuda, costs_cuda, rand_states_cuda, selected_views_cuda,
  // params); CUDA_SAFE_CALL(cudaDeviceSynchronize());

  /*
  for (int i = 0; i < max_iterations; ++i) {
      BlackPixelUpdate<<<grid_size_checkerboard,
  block_size_checkerboard>>>(texture_objects_cuda, texture_depths_cuda,
  cameras_cuda, plane_hypotheses_cuda, costs_cuda, rand_states_cuda,
  selected_views_cuda, params, i); CUDA_SAFE_CALL(cudaDeviceSynchronize());
      RedPixelUpdate<<<grid_size_checkerboard,
  block_size_checkerboard>>>(texture_objects_cuda, texture_depths_cuda,
  cameras_cuda, plane_hypotheses_cuda, costs_cuda, rand_states_cuda,
  selected_views_cuda, params, i); CUDA_SAFE_CALL(cudaDeviceSynchronize());
      printf("iteration: %d\n", i);
  }

  GetDepthandNormal<<<grid_size_randinit, block_size_randinit>>>(cameras_cuda,
  plane_hypotheses_cuda, params); CUDA_SAFE_CALL(cudaDeviceSynchronize());

  BlackPixelFilter<<<grid_size_checkerboard,
  block_size_checkerboard>>>(cameras_cuda, plane_hypotheses_cuda, costs_cuda);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  RedPixelFilter<<<grid_size_checkerboard,
  block_size_checkerboard>>>(cameras_cuda, plane_hypotheses_cuda, costs_cuda);
  CUDA_SAFE_CALL(cudaDeviceSynchronize());
  */

  // cudaMemcpy(plane_hypotheses_host, plane_hypotheses_cuda, sizeof(float4) *
  // width * height, cudaMemcpyDeviceToHost); cudaMemcpy(costs_host,
  // costs_cuda, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
  // CUDA_SAFE_CALL(cudaDeviceSynchronize());
  this->plane_hypotheses_host = tensors.plane_hypotheses_tensor->vector();
  this->costs_host = tensors.costs_tensor->vector();

  // TODO: OUTPUT IMAGE
}

// helpers
void ACMH::SetGeomConsistencyParams() {
  params.geom_consistency = true;
  params.max_iterations = 2;
}

int ACMH::GetReferenceImageWidth() { return sfm.cameras[0].width; }

int ACMH::GetReferenceImageHeight() { return sfm.cameras[0].height; }

// TODO: 90% chance this is wrong
glm::vec4 ACMH::GetPlaneHypothesis(const int index) {
  return *reinterpret_cast<glm::vec4 *>(plane_hypotheses_host.data() + (index * sizeof(glm::vec4)));
}

float ACMH::GetCost(const int index) { return costs_host[index]; }
