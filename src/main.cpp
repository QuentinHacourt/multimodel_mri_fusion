#include "fusion/FusionFactory.h"
#include "io/io.h"
#include "metrics/ssim.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

cv::Mat weightedAverage(std::vector<cv::Mat> &images,
                        std::vector<float> weights);
std::vector<float> normalize(std::vector<float> weights);

int main() {
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

    std::vector<float> weights = {0.1, 0.3, 0.3, 0.3};

    auto ssim = StructuralSimilarityIndexMeasure();

    auto averages =
        FusionFactory::create(FusionFactory::Type::WeightedAverage, weights);

    auto PCA = FusionFactory::create(FusionFactory::Type::PrincipalComponents);

    auto Wavelets = FusionFactory::create(FusionFactory::Type::Wavelet);

    auto Laplace = FusionFactory::create(FusionFactory::Type::Laplacian);

    if (averages) {
        cv::Mat result = averages->fuse(images);

        auto m = ssim.metric(images, result);
        std::cout << m << std::endl;

        showImage(result, "averages");
    } else {
        std::cerr << "Error: invalid averages strategy!" << std::endl;
    }

    if (PCA) {
        cv::Mat result = PCA->fuse(images);

        auto m = ssim.metric(images, result);
        std::cout << m << std::endl;

        showImage(result, "PCA");
    } else {
        std::cerr << "Error: invalid PCA strategy!" << std::endl;
    }

    if (Wavelets) {
        cv::Mat result = Wavelets->fuse(images);

        auto m = ssim.metric(images, result);
        std::cout << m << std::endl;

        showImage(result, "Wavelets");
    } else {
        std::cerr << "Error: invalid wavelets strategy!" << std::endl;
    }

    if (Laplace) {
        cv::Mat result = Laplace->fuse(images);

        auto m = ssim.metric(images, result);
        std::cout << m << std::endl;

        showImage(result, "Laplace");
    } else {
        std::cerr << "Error: invalid laplacian strategy!" << std::endl;
    }

    return 0;
}
