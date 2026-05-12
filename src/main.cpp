#include "fusion/FusionFactory.h"
#include "io/io.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat weightedAverage(std::vector<cv::Mat> &images,
                        std::vector<float> weights);
std::vector<float> normalize(std::vector<float> weights);

int main() {
    int rows = 400;
    int cols = 400;
    std::cout << "hello" << std::endl;

    std::vector<cv::Mat> images = {
        loadImage("data/BraTS2021_00495_t1.nii.gz"),
        loadImage("data/BraTS2021_00495_t1ce.nii.gz"),
        loadImage("data/BraTS2021_00495_t2.nii.gz"),
        loadImage("data/BraTS2021_00495_flair.nii.gz"),
    };

    showImage(images[0], "image 1");
    showImage(images[1], "image 2");
    showImage(images[2], "image 3");
    showImage(images[3], "image 4");

    std::vector<float> weights = {1, 1, 1, 1};

    auto averages =
        FusionFactory::create(FusionFactory::Type::WeightedAverage, weights);

    auto PCA = FusionFactory::create(FusionFactory::Type::PrincipalComponents);

    if (averages) {
        cv::Mat result = averages->fuse(images);

        showImage(result, "averages");
    } else {
        std::cerr << "Error: invalid averages strategy!" << std::endl;
    }

    if (PCA) {
        cv::Mat result = PCA->fuse(images);
        showImage(result, "PCA");
        showImage(result - images[0], "difference");
    } else {
        std::cerr << "Error: invalid PCA strategy!" << std::endl;
    }

    return 0;
}
