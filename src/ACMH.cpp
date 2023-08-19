#include "ACMH.hpp"
#include "helpers.hpp"

#include <glm/vec3.hpp>
#include <filesystem>
#include <sail-common/common.h>
#include <string>
#include <iostream>
namespace fs = std::filesystem;

ACMH::ACMH(std::string img_folder) {
  std::vector<sail::image> images;
  size_t totalSize = 0;
  std::vector<float> combinedBuffer;

  // go through all images
  for (auto path : fs::directory_iterator(img_folder)) {
    sail::image image = sail::image(path.path());
    nr_images++;

    std::vector<float> float_img = helpers::BPP24_RGB_to_float_rgb(image);
    combinedBuffer.insert(combinedBuffer.end(), float_img.begin(), float_img.end());
    break; // TODO: remove
  }

  this->pixels = combinedBuffer;
}

void ACMH::RunPatchMatch() {
    const int width = sfm.cameras[0].width;
    const int height = sfm.cameras[0].height;

    int BLOCK_W = 32;
    int BLOCK_H = (BLOCK_W / 2);

    glm::uvec3 grid_size_randinit;
    grid_size_randinit.x = (width + 16 - 1) / 16;
    grid_size_randinit.y=(height + 16 - 1) / 16;
    grid_size_randinit.z = 1;
    glm::uvec3 block_size_randinit;
    block_size_randinit.x = 16;
    block_size_randinit.y = 16;
    block_size_randinit.z = 1;
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

    kp::Manager mgr;

    // Input tensor
    std::shared_ptr<kp::TensorT<float>> tensorIn = mgr.tensorT(this->pixels);
    auto tensorOut = mgr.tensorT<float>(std::vector<float>(tensorIn->size(), 0x0000FF00));

    std::vector<std::shared_ptr<kp::Tensor>> params = {tensorIn, tensorOut};

    // load shader
    std::ifstream shaderfile("../src/shaders/shader.comp");
    std::stringstream shader;
    shader << shaderfile.rdbuf();

    auto algorithm = mgr.algorithm(
        params, helpers::compileSource(shader.str()), kp::Workgroup({grid_size_randinit.x, grid_size_randinit.y, grid_size_randinit.z}),
        {3024., 4032., 3.},
        std::vector<float>({}));

    // 4. Run operation synchronously using sequence
    mgr.sequence()
        ->record<kp::OpTensorSyncDevice>(params)
        ->record<kp::OpAlgoDispatch>(algorithm) // Binds default push consts
        ->record<kp::OpTensorSyncLocal>(params)
        ->eval();

    // RandomInitialization<<<grid_size_randinit, block_size_randinit>>>(texture_objects_cuda, cameras_cuda, plane_hypotheses_cuda, costs_cuda, rand_states_cuda, selected_views_cuda, params);
    // CUDA_SAFE_CALL(cudaDeviceSynchronize());

    /*
    for (int i = 0; i < max_iterations; ++i) {
        BlackPixelUpdate<<<grid_size_checkerboard, block_size_checkerboard>>>(texture_objects_cuda, texture_depths_cuda, cameras_cuda, plane_hypotheses_cuda, costs_cuda, rand_states_cuda, selected_views_cuda, params, i);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        RedPixelUpdate<<<grid_size_checkerboard, block_size_checkerboard>>>(texture_objects_cuda, texture_depths_cuda, cameras_cuda, plane_hypotheses_cuda, costs_cuda, rand_states_cuda, selected_views_cuda, params, i);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        printf("iteration: %d\n", i);
    }

    GetDepthandNormal<<<grid_size_randinit, block_size_randinit>>>(cameras_cuda, plane_hypotheses_cuda, params);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    BlackPixelFilter<<<grid_size_checkerboard, block_size_checkerboard>>>(cameras_cuda, plane_hypotheses_cuda, costs_cuda);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    RedPixelFilter<<<grid_size_checkerboard, block_size_checkerboard>>>(cameras_cuda, plane_hypotheses_cuda, costs_cuda);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    */

    // cudaMemcpy(plane_hypotheses_host, plane_hypotheses_cuda, sizeof(float4) *
    // width * height, cudaMemcpyDeviceToHost); cudaMemcpy(costs_host,
    // costs_cuda, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
    // CUDA_SAFE_CALL(cudaDeviceSynchronize());
    auto out_pixels = helpers::float_rgb_to_BPP24_RGB(tensorOut->data(), 3024*4032*3);
    sail::image imageOut = sail::image(out_pixels.data(), SAIL_PIXEL_FORMAT_BPP24_RGB, width, height);

    sail::image_output image_output("./out.png");
    image_output.next_frame(imageOut);
    image_output.finish();
}
