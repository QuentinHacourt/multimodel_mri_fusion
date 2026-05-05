#include "fusion/FusionFactory.h"
#include "opencv2/core/mat.hpp"
#include <SimpleITK.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

void showImage(const cv::Mat &img, const std::string &title);
cv::Mat weightedAverage(std::vector<cv::Mat> &images,
                        std::vector<float> weights);
std::vector<float> normalize(std::vector<float> weights);
cv::Mat loadImage(const std::string &imagePath);

namespace sitk = itk::simple;
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
    //
    //
    cv::Mat image = loadImage("../data/BraTS2021_00495_t1ce.nii.gz");
    showImage(image, "image");
}

cv::Mat loadImage(const std::string &imagePath) {
    sitk::Image vol = sitk::ReadImage("../data/BraTS2021_00495_t1ce.nii.gz");

    // 1. Extract the middle slice (e.g., slice 75 of the Z-axis)
    std::vector<unsigned int> size = vol.GetSize();
    std::vector<int> start = {0, 0, (int)size[2] / 2};
    std::vector<unsigned int> sliceSize = {size[0], size[1],
                                           0}; // 0 in Z extracts a 2D slice
    sitk::Image slice = sitk::Extract(vol, sliceSize, start);

    // 2. Rescale intensity to 0-255 and cast to 8-bit for OpenCV
    slice = sitk::RescaleIntensity(slice, 0, 255);
    slice = sitk::Cast(slice, sitk::sitkUInt8);

    // 3. Convert to OpenCV Mat
    // SimpleITK image buffer is contiguous.
    cv::Mat cvImage(cv::Size(slice.GetWidth(), slice.GetHeight()), CV_8UC1,
                    slice.GetBufferAsUInt8());

    // 4. IMPORTANT: Clone the mat if you plan to use it after 'slice' goes out
    // of scope
    cv::Mat result = cvImage.clone();
    return result;
}

void showImage(const cv::Mat &img, const std::string &title) {
    if (img.empty()) {
        std::cerr << "ERROR: Can't show empty image" << std::endl;
    }

    cv::imshow(title, img);
    cv::waitKey(0);
}
