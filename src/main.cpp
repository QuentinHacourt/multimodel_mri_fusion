#include "fusion/FusionFactory.h"
#include "io/io.h"
#include <SimpleITK.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat weightedAverage(std::vector<cv::Mat> &images,
                        std::vector<float> weights);
std::vector<float> normalize(std::vector<float> weights);

int main() {
    // int rows = 400;
    // int cols = 400;

    // std::vector<cv::Mat> images = {
    //     cv::Mat(rows, cols, CV_32F, cv::Scalar(1.0f)),
    //     cv::Mat(rows, cols, CV_32F, cv::Scalar(0.0f)),
    //     cv::Mat(rows, cols, CV_32F, cv::Scalar(0.5f)),
    //     cv::Mat(rows, cols, CV_32F, cv::Scalar(0.75f))};

    // std::vector<float> weights = {1, 1, 1, 1};

    // auto strategy =
    //     FusionFactory::create(FusionFactory::Type::WeightedAverage, weights);

    // if (strategy) {
    //     cv::Mat result = strategy->fuse(images);

    //     showImage(images[0], "image 1");
    //     showImage(images[1], "image 2");
    //     showImage(images[2], "image 3");
    //     showImage(images[3], "image 4");
    //     showImage(result, "result");
    // } else {
    //     std::cerr << "Error: invalid strategy!" << std::endl;
    // }

    // return 0;
    cv::Mat image = loadImage("data/BraTS2021_00495_t1ce.nii.gz");
    showImage(image, "image");
}
