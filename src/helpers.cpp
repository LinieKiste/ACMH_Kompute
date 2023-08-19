#include "helpers.hpp"

namespace helpers {
std::vector<uint32_t> compileSource(const std::string &source) {
  std::ofstream fileOut("tmp_kp_shader.comp");
  fileOut << source;
  fileOut.close();
  if (system(std::string("glslangValidator -V tmp_kp_shader.comp -o "
                         "tmp_kp_shader.comp.spv")
                 .c_str()))
    throw std::runtime_error("Error running glslangValidator command");
  std::ifstream fileStream("tmp_kp_shader.comp.spv", std::ios::binary);
  std::vector<char> buffer;
  buffer.insert(buffer.begin(), std::istreambuf_iterator<char>(fileStream), {});
  return {(uint32_t *)buffer.data(),
          (uint32_t *)(buffer.data() + buffer.size())};
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

std::vector<uint8_t> float_rgb_to_BPP24_RGB(void *data, size_t size) {
  std::vector<uint8_t> result;
  result.reserve(size * (3 / 4));

  float *float_data = (float *)data;
  for (int i = 0; i < size; i++) {
    if (i % 4 == 3)
      continue; // skip last pixel

    auto new_pixel = static_cast<uint8_t>(float_data[i] * 255);
    result.emplace_back(new_pixel);
  }
  return result;
}
} // namespace helpers