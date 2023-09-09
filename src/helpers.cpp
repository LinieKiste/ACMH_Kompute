#include "helpers.hpp"
#include "Tracy.hpp"

namespace helpers {
std::vector<uint32_t> loadShader(const std::string &name) {
    std::stringstream in_path, spv_path, command;
    in_path << "./src/shaders/" << name;
    spv_path << "./spv/" << name << ".spv";
    // add -gVS to glslangValidator to get debug symbols
    command << "glslangValidator --quiet -V " << in_path.str() << " -o " << spv_path.str();

    if (system(command.str().c_str())) {
      throw std::runtime_error("Error running glslangValidator command");
    }

    std::ifstream fileStream(spv_path.str(), std::ios::binary);
    std::vector<char> buffer;
    buffer.insert(buffer.begin(), std::istreambuf_iterator<char>(fileStream),
                  {});
    return {(uint32_t *)buffer.data(),
            (uint32_t *)(buffer.data() + buffer.size())};
}

// file helpers
int readDepthDmb(const std::string file_path, cv::Mat_<float> &depth)
{
    FILE *inimage;
    inimage = fopen(file_path.c_str(), "rb");
    if (!inimage){
        std::cout << "Error opening file " << file_path << std::endl;
        return -1;
    }

    int32_t type, h, w, nb;

    type = -1;

    fread(&type,sizeof(int32_t),1,inimage);
    fread(&h,sizeof(int32_t),1,inimage);
    fread(&w,sizeof(int32_t),1,inimage);
    fread(&nb,sizeof(int32_t),1,inimage);

    if (type != 1) {
        fclose(inimage);
        return -1;
    }

    int32_t dataSize = h*w*nb;

    depth = cv::Mat::zeros(h,w,CV_32F);
    fread(depth.data,sizeof(float),dataSize,inimage);

    fclose(inimage);
    return 0;
}

int writeDepthDmb(const std::string file_path, const cv::Mat_<float> depth)
{
    FILE *outimage;
    outimage = fopen(file_path.c_str(), "wb");
    if (!outimage) {
        std::cout << "Error opening file " << file_path << std::endl;
    }

    int32_t type = 1;
    int32_t h = depth.rows;
    int32_t w = depth.cols;
    int32_t nb = 1;

    fwrite(&type,sizeof(int32_t),1,outimage);
    fwrite(&h,sizeof(int32_t),1,outimage);
    fwrite(&w,sizeof(int32_t),1,outimage);
    fwrite(&nb,sizeof(int32_t),1,outimage);

    float* data = (float*)depth.data;

    int32_t datasize = w*h*nb;
    fwrite(data,sizeof(float),datasize,outimage);

    fclose(outimage);
    return 0;
}

int readNormalDmb (const std::string file_path, cv::Mat_<cv::Vec3f> &normal)
{
    FILE *inimage;
    inimage = fopen(file_path.c_str(), "rb");
    if (!inimage){
        std::cout << "Error opening file " << file_path << std::endl;
        return -1;
    }

    int32_t type, h, w, nb;

    type = -1;

    fread(&type,sizeof(int32_t),1,inimage);
    fread(&h,sizeof(int32_t),1,inimage);
    fread(&w,sizeof(int32_t),1,inimage);
    fread(&nb,sizeof(int32_t),1,inimage);

    if (type != 1) {
        fclose(inimage);
        return -1;
    }

    int32_t dataSize = h*w*nb;

    normal = cv::Mat::zeros(h,w,CV_32FC3);
    fread(normal.data,sizeof(float),dataSize,inimage);

    fclose(inimage);
    return 0;
}

int writeNormalDmb(const std::string file_path, const cv::Mat_<cv::Vec3f> normal)
{
    FILE *outimage;
    outimage = fopen(file_path.c_str(), "wb");
    if (!outimage) {
        std::cout << "Error opening file " << file_path << std::endl;
    }

    int32_t type = 1;
    int32_t h = normal.rows;
    int32_t w = normal.cols;
    int32_t nb = 3;

    fwrite(&type,sizeof(int32_t),1,outimage);
    fwrite(&h,sizeof(int32_t),1,outimage);
    fwrite(&w,sizeof(int32_t),1,outimage);
    fwrite(&nb,sizeof(int32_t),1,outimage);

    float* data = (float*)normal.data;

    int32_t datasize = w*h*nb;
    fwrite(data,sizeof(float),datasize,outimage);

    fclose(outimage);
    return 0;
}
// file helpers end


void RescaleImageAndCamera(cv::Mat_<cv::Vec3b> &src, cv::Mat_<cv::Vec3b> &dst, cv::Mat_<float> &depth, Camera &camera)
{
    const int cols = depth.cols;
    const int rows = depth.rows;

    if (cols == src.cols && rows == src.rows) {
        dst = src.clone();
        return;
    }

    const float scale_x = cols / static_cast<float>(src.cols);
    const float scale_y = rows / static_cast<float>(src.rows);

    cv::resize(src, dst, cv::Size(cols,rows), 0, 0, cv::INTER_LINEAR);

    camera.K[0] *= scale_x;
    camera.K[2] *= scale_x;
    camera.K[4] *= scale_y;
    camera.K[5] *= scale_y;
    camera.width = cols;
    camera.height = rows;
}

void GenerateSampleList(const std::string &dense_folder, std::vector<Problem> &problems)
{
    std::string cluster_list_path = dense_folder + std::string("/pair.txt");

    problems.clear();

    std::ifstream file(cluster_list_path);
    if(file.fail()) {
        exit(-1);
    }

    int num_images;
    file >> num_images;

    for (int i = 0; i < num_images; ++i) {
        Problem problem;
        problem.src_image_ids.clear();
        file >> problem.ref_image_id;

        int num_src_images;
        file >> num_src_images;
        for (int j = 0; j < num_src_images; ++j) {
            int id;
            float score;
            file >> id >> score;
            if (score <= 0.0f) {
                continue;
            }
            problem.src_image_ids.push_back(id);
        }
        problems.push_back(problem);
    }
}

void ProcessProblem(const std::string &dense_folder, const Problem &problem, bool geom_consistency, Cache cache)
{
    ZoneScoped;
    std::cout << "Processing image " << std::setw(8) << std::setfill('0') << problem.ref_image_id << "..." << std::endl;
    std::stringstream result_path;
    result_path << dense_folder << "/ACMH" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
    std::string result_folder = result_path.str();
    mkdir(result_folder.c_str(), 0777);

    ACMH acmh(dense_folder, problem.ref_image_id, problem.src_image_ids, cache);
    if (geom_consistency) {
      acmh.SetGeomConsistencyParams();
      acmh.init_depths(dense_folder, problem.ref_image_id, problem.src_image_ids);
    }

    acmh.VulkanSpaceInitialization(dense_folder, problem.ref_image_id, problem.src_image_ids);
    acmh.RunPatchMatch();

    const int width = acmh.GetReferenceImageWidth();
    const int height = acmh.GetReferenceImageHeight();

    cv::Mat_<float> depths = cv::Mat::zeros(height, width, CV_32FC1);
    cv::Mat_<cv::Vec3f> normals = cv::Mat::zeros(height, width, CV_32FC3);
    cv::Mat_<float> costs = cv::Mat::zeros(height, width, CV_32FC1);

    for (int col = 0; col < width; ++col) {
        for (int row = 0; row < height; ++row) {
            int center = row * width + col;
            glm::vec4 plane_hypothesis = acmh.GetPlaneHypothesis(center);
            depths(row, col) = plane_hypothesis.w;
            normals(row, col) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
            costs(row, col) = acmh.GetCost(center);
        }
    }

    std::string suffix = "/depths.dmb";
    if (geom_consistency) {
        suffix = "/depths_geom.dmb";
    }
    std::string depth_path = result_folder + suffix;
    std::string normal_path = result_folder + "/normals.dmb";
    std::string cost_path = result_folder + "/costs.dmb";
    writeDepthDmb(depth_path, depths);
    writeNormalDmb(normal_path, normals);
    writeDepthDmb(cost_path, costs);
    std::cout << "Processing image " << std::setw(8) << std::setfill('0') << problem.ref_image_id << " done!" << std::endl;
}

void StoreColorPlyFileBinaryPointCloud (const std::string &plyFilePath, const std::vector<PointList> &pc)
{
    std::cout << "store 3D points to ply file" << std::endl;

    FILE *outputPly;
    outputPly=fopen(plyFilePath.c_str(), "wb");

    /*write header*/
    fprintf(outputPly, "ply\n");
    fprintf(outputPly, "format binary_little_endian 1.0\n");
    fprintf(outputPly, "element vertex %d\n",pc.size());
    fprintf(outputPly, "property float x\n");
    fprintf(outputPly, "property float y\n");
    fprintf(outputPly, "property float z\n");
    fprintf(outputPly, "property float nx\n");
    fprintf(outputPly, "property float ny\n");
    fprintf(outputPly, "property float nz\n");
    fprintf(outputPly, "property uchar red\n");
    fprintf(outputPly, "property uchar green\n");
    fprintf(outputPly, "property uchar blue\n");
    fprintf(outputPly, "end_header\n");

    //write data
#pragma omp parallel for
    for(size_t i = 0; i < pc.size(); i++) {
        const PointList &p = pc[i];
        glm::vec3 X = p.coord;
        const glm::vec3 normal = p.normal;
        const glm::vec3 color = p.color;
        const char b_color = (int)color.x;
        const char g_color = (int)color.y;
        const char r_color = (int)color.z;

        if(!(X.x < FLT_MAX && X.x > -FLT_MAX) || !(X.y < FLT_MAX && X.y > -FLT_MAX) || !(X.z < FLT_MAX && X.z >= -FLT_MAX)){
            X.x = 0.0f;
            X.y = 0.0f;
            X.z = 0.0f;
        }
#pragma omp critical
        {
            fwrite(&X.x,      sizeof(X.x), 1, outputPly);
            fwrite(&X.y,      sizeof(X.y), 1, outputPly);
            fwrite(&X.z,      sizeof(X.z), 1, outputPly);
            fwrite(&normal.x, sizeof(normal.x), 1, outputPly);
            fwrite(&normal.y, sizeof(normal.y), 1, outputPly);
            fwrite(&normal.z, sizeof(normal.z), 1, outputPly);
            fwrite(&r_color,  sizeof(char), 1, outputPly);
            fwrite(&g_color,  sizeof(char), 1, outputPly);
            fwrite(&b_color,  sizeof(char), 1, outputPly);
        }

    }
    fclose(outputPly);
}

glm::vec3 Get3DPointonWorld(const int x, const int y, const float depth, const Camera camera)
{
    glm::vec3 pointX;
    glm::vec3 tmpX;
    // Reprojection
    pointX.x = depth * (x - camera.K[2]) / camera.K[0];
    pointX.y = depth * (y - camera.K[5]) / camera.K[4];
    pointX.z = depth;

    // Rotation
    tmpX.x = camera.R[0] * pointX.x + camera.R[3] * pointX.y + camera.R[6] * pointX.z;
    tmpX.y = camera.R[1] * pointX.x + camera.R[4] * pointX.y + camera.R[7] * pointX.z;
    tmpX.z = camera.R[2] * pointX.x + camera.R[5] * pointX.y + camera.R[8] * pointX.z;

    // Transformation
    glm::vec3 C;
    C.x = -(camera.R[0] * camera.t[0] + camera.R[3] * camera.t[1] + camera.R[6] * camera.t[2]);
    C.y = -(camera.R[1] * camera.t[0] + camera.R[4] * camera.t[1] + camera.R[7] * camera.t[2]);
    C.z = -(camera.R[2] * camera.t[0] + camera.R[5] * camera.t[1] + camera.R[8] * camera.t[2]);
    pointX.x = tmpX.x + C.x;
    pointX.y = tmpX.y + C.y;
    pointX.z = tmpX.z + C.z;

    return pointX;
}

void ProjectonCamera(const glm::vec3 PointX, const Camera camera, glm::vec2 &point, float &depth)
{
    glm::vec3 tmp;
    tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y + camera.R[2] * PointX.z + camera.t[0];
    tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y + camera.R[5] * PointX.z + camera.t[1];
    tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y + camera.R[8] * PointX.z + camera.t[2];

    depth = camera.K[6] * tmp.x + camera.K[7] * tmp.y + camera.K[8] * tmp.z;
    point.x = (camera.K[0] * tmp.x + camera.K[1] * tmp.y + camera.K[2] * tmp.z) / depth;
    point.y = (camera.K[3] * tmp.x + camera.K[4] * tmp.y + camera.K[5] * tmp.z) / depth;
}

float GetAngle( const cv::Vec3f &v1, const cv::Vec3f &v2 )
{
    float dot_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    float angle = acosf(dot_product);
    //if angle is not a number the dot product was 1 and thus the two vectors should be identical --> return 0
    if ( angle != angle )
        return 0.0f;

    return angle;
}

void RunFusion(std::string &dense_folder, const std::vector<Problem> &problems, bool geom_consistency)
{
    ZoneScoped;
    size_t num_images = problems.size();
    std::string image_folder = dense_folder + std::string("/images");
    std::string cam_folder = dense_folder + std::string("/cams");

    std::vector<cv::Mat> images;
    std::vector<Camera> cameras;
    std::vector<cv::Mat_<float>> depths;
    std::vector<cv::Mat_<cv::Vec3f>> normals;
    std::vector<cv::Mat> masks;
    images.clear();
    cameras.clear();
    depths.clear();
    normals.clear();
    masks.clear();
    
    std::map<int, int> image_id_2_index;

    for (size_t i = 0; i < num_images; ++i) {
        std::cout << "Reading image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
        image_id_2_index[problems[i].ref_image_id] = i;
        std::stringstream image_path;
        image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << ".jpg";
        cv::Mat_<cv::Vec3b> image = cv::imread (image_path.str(), cv::IMREAD_COLOR);
        std::stringstream cam_path;
        cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id << "_cam.txt";
        Camera camera = SfM::ReadCamera(cam_path.str());

        std::stringstream result_path;
        result_path << dense_folder << "/ACMH" << "/2333_" << std::setw(8) << std::setfill('0') << problems[i].ref_image_id;
        std::string result_folder = result_path.str();
        std::string suffix = "/depths.dmb";
        if (geom_consistency) {
            suffix = "/depths_geom.dmb";
        }
        std::string depth_path = result_folder + suffix;
        std::string normal_path = result_folder + "/normals.dmb";
        cv::Mat_<float> depth;
        cv::Mat_<cv::Vec3f> normal;
        readDepthDmb(depth_path, depth);
        readNormalDmb(normal_path, normal);

        cv::Mat_<cv::Vec3b> scaled_image;
        RescaleImageAndCamera(image, scaled_image, depth, camera);
        images.push_back(scaled_image);
        cameras.push_back(camera);
        depths.push_back(depth);
        normals.push_back(normal);
        cv::Mat mask = cv::Mat::zeros(depth.rows, depth.cols, CV_8UC1);
        masks.push_back(mask);
    }

    std::vector<PointList> PointCloud;
    PointCloud.clear();

    for (size_t i = 0; i < num_images; ++i) {
        std::cout << "Fusing image " << std::setw(8) << std::setfill('0') << i << "..." << std::endl;
        const int cols = depths[i].cols;
        const int rows = depths[i].rows;
        int num_ngb = problems[i].src_image_ids.size();
        std::vector<glm::ivec2> used_list(num_ngb, glm::ivec2(-1, -1));
        for (int r =0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (masks[i].at<uchar>(r, c) == 1)
                    continue;
                float ref_depth = depths[i].at<float>(r, c);
                cv::Vec3f ref_normal = normals[i].at<cv::Vec3f>(r, c);

                if (ref_depth <= 0.0)
                    continue;

                glm::vec3 PointX = Get3DPointonWorld(c, r, ref_depth, cameras[i]);
                glm::vec3 consistent_Point = PointX;
                cv::Vec3f consistent_normal = ref_normal;
                float consistent_Color[3] = {(float)images[i].at<cv::Vec3b>(r, c)[0], (float)images[i].at<cv::Vec3b>(r, c)[1], (float)images[i].at<cv::Vec3b>(r, c)[2]};
                int num_consistent = 0;

                for (int j = 0; j < num_ngb; ++j) {
                    int src_id = image_id_2_index[problems[i].src_image_ids[j]];
                    const int src_cols = depths[src_id].cols;
                    const int src_rows = depths[src_id].rows;
                    glm::vec2 point;
                    float proj_depth;
                    ProjectonCamera(PointX, cameras[src_id], point, proj_depth);
                    int src_r = int(point.y + 0.5f);
                    int src_c = int(point.x + 0.5f);
                    if (src_c >= 0 && src_c < src_cols && src_r >= 0 && src_r < src_rows) {
                        if (masks[src_id].at<uchar>(src_r, src_c) == 1)
                            continue;

                        float src_depth = depths[src_id].at<float>(src_r, src_c);
                        cv::Vec3f src_normal = normals[src_id].at<cv::Vec3f>(src_r, src_c);
                        if (src_depth <= 0.0)
                            continue;

                        glm::vec3 tmp_X = Get3DPointonWorld(src_c, src_r, src_depth, cameras[src_id]);
                        glm::vec2 tmp_pt;
                        ProjectonCamera(tmp_X, cameras[i], tmp_pt, proj_depth);
                        float reproj_error = sqrt(pow(c - tmp_pt.x, 2) + pow(r - tmp_pt.y, 2));
                        float relative_depth_diff = fabs(proj_depth - ref_depth) / ref_depth;
                        float angle = GetAngle(ref_normal, src_normal);

                        if (reproj_error < 2.0f && relative_depth_diff < 0.01f && angle < 0.174533f) {
                            consistent_Point.x += tmp_X.x;
                            consistent_Point.y += tmp_X.y;
                            consistent_Point.z += tmp_X.z;
                            consistent_normal = consistent_normal + src_normal;
                            consistent_Color[0] += images[src_id].at<cv::Vec3b>(src_r, src_c)[0];
                            consistent_Color[1] += images[src_id].at<cv::Vec3b>(src_r, src_c)[1];
                            consistent_Color[2] += images[src_id].at<cv::Vec3b>(src_r, src_c)[2];

                            used_list[j].x = src_c;
                            used_list[j].y = src_r;
                            num_consistent++;
                        }
                    }
                }

                if (num_consistent >= 2) {
                    consistent_Point.x /= (num_consistent + 1.0f);
                    consistent_Point.y /= (num_consistent + 1.0f);
                    consistent_Point.z /= (num_consistent + 1.0f);
                    consistent_normal /= (num_consistent + 1.0f);
                    consistent_Color[0] /= (num_consistent + 1.0f);
                    consistent_Color[1] /= (num_consistent + 1.0f);
                    consistent_Color[2] /= (num_consistent + 1.0f);

                    PointList point3D;
                    point3D.coord = consistent_Point;
                    point3D.normal = glm::vec3(consistent_normal[0], consistent_normal[1], consistent_normal[2]);
                    point3D.color = glm::vec3(consistent_Color[0], consistent_Color[1], consistent_Color[2]);
                    PointCloud.push_back(point3D);

                    for (int j = 0; j < num_ngb; ++j) {
                        if (used_list[j].x == -1)
                            continue;
                        masks[image_id_2_index[problems[i].src_image_ids[j]]].at<uchar>(used_list[j].y, used_list[j].x) = 1;
                    }
                }
            }
        }
    }

    std::string ply_path = dense_folder + "/ACMH/ACMH_model.ply";
    StoreColorPlyFileBinaryPointCloud (ply_path, PointCloud);
}

} // namespace helpers