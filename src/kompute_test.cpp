#include "kompute/Core.hpp"
#include "kompute/Kompute.hpp"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <sail-common/common.h>
#include <sail-common/pixel.h>
#include <vector>
#include "kompute/operations/OpTensorSyncLocal.hpp"
#include "sail-c++/sail-c++.h"

static
std::vector<uint32_t>
compileSource(
  const std::string& source)
{
    std::ofstream fileOut("tmp_kp_shader.comp");
    fileOut << source;
    fileOut.close();
    if (system(std::string("glslangValidator -V tmp_kp_shader.comp -o tmp_kp_shader.comp.spv").c_str()))
        throw std::runtime_error("Error running glslangValidator command");
    std::ifstream fileStream("tmp_kp_shader.comp.spv", std::ios::binary);
    std::vector<char> buffer;
    buffer.insert(buffer.begin(), std::istreambuf_iterator<char>(fileStream), {});
    return {(uint32_t*)buffer.data(), (uint32_t*)(buffer.data() + buffer.size())};
}

std::vector<float> BPP24_RGB_to_float_rgb(sail::image input) {
    std::vector<float> result;
    result.reserve(input.pixels_size());

    uint8_t* input_pixels = (uint8_t*)input.pixels();
    for(int i = 0; i < input.pixels_size(); i++) {
        auto pixel = static_cast<float>(input_pixels[i])/255;
        result.emplace_back(pixel);
        if(i%3 == 2) {
            result.emplace_back(pixel);
        }
    }

    return result;
}

std::vector<uint8_t> float_rgb_to_BPP24_RGB(void* data, size_t size) {
    std::vector<uint8_t> result;
    result.reserve(size * (3/4));

    float* float_data = (float*)data;
    for(int i=0; i<size; i++) {
        if(i%4 == 3) continue; // skip last pixel

        auto new_pixel = static_cast<uint8_t>(float_data[i]*255);
        result.emplace_back(new_pixel);
    }
    return result;
}

int kompute(sail::image &image, const std::string& shader) {
    auto width = image.width();
    auto height = image.height();

    // 1. Create Kompute Manager with default settings (device 0, first queue and no extensions)
    kp::Manager mgr;

    // 2. Create and initialise Kompute Tensors through manager

    // Input tensor
    assert(image.pixel_format() == SAIL_PIXEL_FORMAT_BPP24_RGB);
    auto float_vec = BPP24_RGB_to_float_rgb(image);
    std::shared_ptr<kp::TensorT<float>> tensorIn = mgr.tensorT(float_vec);

    // output Tensor
    auto tensorOut = mgr.tensorT<float>(std::vector<float>(tensorIn->size(), 0x0000FF00));

    std::vector<std::shared_ptr<kp::Tensor>> params = {tensorIn, tensorOut};

    // 3. Create algorithm based on shader (supports buffers & push/spec constants)
    auto algorithm = mgr.algorithm(
        params, compileSource(shader), kp::Workgroup({width, height, 1}),
        {static_cast<float>(image.width()), static_cast<float>(image.height()),
         static_cast<float>(image.palette().color_count())},
        std::vector<float>({}));

    // 4. Run operation synchronously using sequence
    mgr.sequence()
        ->record<kp::OpTensorSyncDevice>(params)
        ->record<kp::OpAlgoDispatch>(algorithm) // Binds default push consts
        ->record<kp::OpTensorSyncLocal>(params)
        ->eval();

    auto out_pixels = float_rgb_to_BPP24_RGB(tensorOut->data(), tensorOut->size());
    sail::image imageOut = sail::image(out_pixels.data(), image.pixel_format(), width, height);

    image = imageOut;
    return 0;
} // Manages / releases all CPU and GPU memory resources
