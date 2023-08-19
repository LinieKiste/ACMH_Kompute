#include "sfm.hpp"

void SfM::InputInitialization(const std::string &dense_folder,
                          const Problem &problem) {
  images.clear();
  cameras.clear();

  std::string image_folder = dense_folder + std::string("/images");
  std::string cam_folder = dense_folder + std::string("/cams");

  std::stringstream image_path;
  image_path << image_folder << "/" << std::setw(8) << std::setfill('0')
             << problem.ref_image_id << ".jpg";
  cv::Mat_<uint8_t> image_uint =
      cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
  cv::Mat image_float;
  image_uint.convertTo(image_float, CV_32FC1);
  images.push_back(image_float);
  std::stringstream cam_path;
  cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0')
           << problem.ref_image_id << "_cam.txt";
  Camera camera = ReadCamera(cam_path.str());
  camera.height = image_float.rows;
  camera.width = image_float.cols;
  cameras.push_back(camera);

  size_t num_src_images = problem.src_image_ids.size();
  for (size_t i = 0; i < num_src_images; ++i) {
    std::stringstream image_path;
    image_path << image_folder << "/" << std::setw(8) << std::setfill('0')
               << problem.src_image_ids[i] << ".jpg";
    cv::Mat_<uint8_t> image_uint =
        cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
    cv::Mat image_float;
    image_uint.convertTo(image_float, CV_32FC1);
    images.push_back(image_float);
    std::stringstream cam_path;
    cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0')
             << problem.src_image_ids[i] << "_cam.txt";
    Camera camera = ReadCamera(cam_path.str());
    camera.height = image_float.rows;
    camera.width = image_float.cols;
    cameras.push_back(camera);
  }

  // Scale cameras and images
  for (size_t i = 0; i < images.size(); ++i) {
    if (images[i].cols <= params.max_image_size &&
        images[i].rows <= params.max_image_size) {
      continue;
    }

    const float factor_x =
        static_cast<float>(params.max_image_size) / images[i].cols;
    const float factor_y =
        static_cast<float>(params.max_image_size) / images[i].rows;
    const float factor = std::min(factor_x, factor_y);

    const int new_cols = std::round(images[i].cols * factor);
    const int new_rows = std::round(images[i].rows * factor);

    const float scale_x = new_cols / static_cast<float>(images[i].cols);
    const float scale_y = new_rows / static_cast<float>(images[i].rows);

    cv::Mat_<float> scaled_image_float;
    cv::resize(images[i], scaled_image_float, cv::Size(new_cols, new_rows), 0,
               0, cv::INTER_LINEAR);
    images[i] = scaled_image_float.clone();

    cameras[i].K[0] *= scale_x;
    cameras[i].K[2] *= scale_x;
    cameras[i].K[4] *= scale_y;
    cameras[i].K[5] *= scale_y;
    cameras[i].height = scaled_image_float.rows;
    cameras[i].width = scaled_image_float.cols;
  }

  params.depth_min = cameras[0].depth_min * 0.6f;
  params.depth_max = cameras[0].depth_max * 1.2f;
  std::cout << "depthe range: " << params.depth_min << " " << params.depth_max
            << std::endl;
  params.num_images = (int)images.size();
  std::cout << "num images: " << params.num_images << std::endl;
  params.disparity_min = cameras[0].K[0] * params.baseline / params.depth_max;
  params.disparity_max = cameras[0].K[0] * params.baseline / params.depth_min;

  if (params.geom_consistency) {
    depths.clear();

    std::stringstream result_path;
    result_path << dense_folder << "/ACMH"
                << "/2333_" << std::setw(8) << std::setfill('0')
                << problem.ref_image_id;
    std::string result_folder = result_path.str();
    std::string depth_path = result_folder + "/depths.dmb";
    cv::Mat_<float> ref_depth;
    readDepthDmb(depth_path, ref_depth);
    depths.push_back(ref_depth);

    size_t num_src_images = problem.src_image_ids.size();
    for (size_t i = 0; i < num_src_images; ++i) {
      std::stringstream result_path;
      result_path << dense_folder << "/ACMH"
                  << "/2333_" << std::setw(8) << std::setfill('0')
                  << problem.src_image_ids[i];
      std::string result_folder = result_path.str();
      std::string depth_path = result_folder + "/depths.dmb";
      cv::Mat_<float> depth;
      readDepthDmb(depth_path, depth);
      depths.push_back(depth);
    }
  }
}
