#include "main.hpp"

int main(int argc, char** argv) {
  if (argc < 2){
        std::cout << "USAGE: ACMH dense_folder" << std::endl;
        return -1;
  }

    std::string dense_folder = argv[1];
    std::vector<Problem> problems;
    helpers::GenerateSampleList(dense_folder, problems);

    std::string output_folder = dense_folder + std::string("/ACMH");
    mkdir(output_folder.c_str(), 0777);

    size_t num_images = problems.size();
    std::cout << "There are " << num_images << " problems needed to be processed!" << std::endl;

    bool geom_consistency = false;
    for (size_t i = 0; i < num_images; ++i) {
        helpers::ProcessProblem(dense_folder, problems[i], geom_consistency);
        // TODO: remove this
        break;
    }
    return 0;
    geom_consistency = true;
    for (size_t i = 0; i < num_images; ++i) {
        helpers::ProcessProblem(dense_folder, problems[i], geom_consistency);
    }

    // RunFusion(dense_folder, problems, geom_consistency);

    return 0;
}
