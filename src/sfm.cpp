#include "sfm.hpp"

SfM::SfM(const std::string &dense_folder, int ref_image_id, std::vector<int> src_image_ids) {
  images.clear();
  cameras.clear();

  std::string image_folder = dense_folder + std::string("/images");
  std::string cam_folder = dense_folder + std::string("/cams");

  // reference image and camera
  get_image_and_camera(ref_image_id, image_folder, cam_folder);

  // src images and cameras
  size_t num_src_images = src_image_ids.size();
  for (size_t i = 0; i < num_src_images; ++i) {
    get_image_and_camera(src_image_ids[i], image_folder, cam_folder);
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
  std::cout << "depth range: " << params.depth_min << " " << params.depth_max
            << std::endl;
  params.num_images = (int)images.size();
  std::cout << "num images: " << params.num_images << std::endl;
  params.disparity_min = cameras[0].K[0] * params.baseline / params.depth_max;
  params.disparity_max = cameras[0].K[0] * params.baseline / params.depth_min;
}

Camera SfM::ReadCamera(const std::string &cam_path)
{
    Camera camera;
    std::ifstream file(cam_path);

    std::string line;
    file >> line;

    for (int i = 0; i < 3; ++i) {
        file >> camera.R[3 * i + 0] >> camera.R[3 * i + 1] >> camera.R[3 * i + 2] >> camera.t[i];
    }

    float tmp[4];
    file >> tmp[0] >> tmp[1] >> tmp[2] >> tmp[3];
    file >> line;

    for (int i = 0; i < 3; ++i) {
        file >> camera.K[3 * i + 0] >> camera.K[3 * i + 1] >> camera.K[3 * i + 2];
    }

    float depth_num;
    float interval;
    file >> camera.depth_min >> interval >> depth_num >> camera.depth_max;

    return camera;
}

void SfM::get_image_and_camera(int image_id, std::string image_folder, std::string cam_folder){
    std::stringstream image_path;
    image_path << image_folder << "/" << std::setw(8) << std::setfill('0')
               << image_id << ".jpg";
    cv::Mat_<uint8_t> image_uint =
        cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
    cv::Mat image_float;
    image_uint.convertTo(image_float, CV_32FC1);
    images.push_back(image_float);
    std::stringstream cam_path;
    cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0')
             << image_id << "_cam.txt";
    Camera camera = ReadCamera(cam_path.str());
    camera.height = image_float.rows;
    camera.width = image_float.cols;
    cameras.push_back(camera);
}
