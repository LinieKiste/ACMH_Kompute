#include "kompute_test.cpp"
#include <cstdlib>
#include <fstream>
#include <sail-c++/sail-c++.h>
#include <string>

int main() {
  // load image
  sail::image image = sail::image("../../Bilder/IMG_6723.png");

  // load shader
  std::ifstream shaderfile("../src/shaders/shader.comp");
  std::stringstream shader;
  shader << shaderfile.rdbuf();

  kompute(image, shader.str());

  sail::image_output image_output("./out.png");
  image_output.next_frame(image);
  image_output.finish();
}
