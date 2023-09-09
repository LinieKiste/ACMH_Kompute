#include "cache.hpp"
#include "helpers.hpp"

Cache::Cache() {
  // load shaders
  shaders.random_init = helpers::loadShader("random_init.comp");
  shaders.black_pixel_update = helpers::loadShader("black_pixel_update.comp");
  shaders.red_pixel_update = helpers::loadShader("red_pixel_update.comp");
  shaders.get_depth_and_normal = helpers::loadShader("get_depth_normal.comp");
  shaders.black_filter = helpers::loadShader("black_filter.comp");
  shaders.red_filter = helpers::loadShader("red_filter.comp");
};
