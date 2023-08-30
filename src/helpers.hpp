#pragma once
#include "ACMH.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <vector>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>

#include <sys/stat.h> // mkdir
#include <sys/types.h> // mkdir

struct Problem {
    int ref_image_id;
    std::vector<int> src_image_ids;
};

namespace helpers {
std::vector<uint32_t> compileSource(const std::string &source);

int readDepthDmb(const std::string file_path, cv::Mat_<float> &depth);
int writeDepthDmb(const std::string file_path, const cv::Mat_<float> depth);
int readNormalDmb (const std::string file_path, cv::Mat_<cv::Vec3f> &normal);
int writeNormalDmb(const std::string file_path, const cv::Mat_<cv::Vec3f> normal);

void ProcessProblem(const std::string &dense_folder, const Problem &problem, bool geom_consistency);
void GenerateSampleList(const std::string &dense_folder, std::vector<Problem> &problems);
}
