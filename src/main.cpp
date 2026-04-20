#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

void showImage(const cv::Mat &img, const std::string &title);
cv::Mat weightedAverage(std::vector<cv::Mat> &images,
                        std::vector<float> weights);
std::vector<float> normalize(std::vector<float> weights);

int main() {
  int rows = 400;
  int cols = 400;

  std::vector<cv::Mat> images = {
      cv::Mat(rows, cols, CV_32F, cv::Scalar(1.0f)),
      cv::Mat(rows, cols, CV_32F, cv::Scalar(0.0f)),
      cv::Mat(rows, cols, CV_32F, cv::Scalar(0.5f)),
      cv::Mat(rows, cols, CV_32F, cv::Scalar(0.75f))};

  std::vector<float> weights = {1, 1, 1, 1};

  cv::Mat imageFused = weightedAverage(images, weights);

  showImage(images[0], "image 1");
  showImage(images[1], "image 2");
  showImage(images[2], "image 3");
  showImage(images[3], "image 4");
  showImage(imageFused, "image fused");
  return 0;
}

cv::Mat weightedAverage(std::vector<cv::Mat> &images,
                        std::vector<float> weights) {
  cv::Mat res = cv::Mat::zeros(images[0].size(), images[0].type());

  weights = normalize(weights);

  for (int i = 0; i < images.size(); i++)
    res += images[i] * weights[i];

  return res;
}

std::vector<float> normalize(std::vector<float> weights) {
  float sum = 0;
  for (float weight : weights)
    sum += weight;

  for (float &weight : weights)
    weight /= sum;

  return weights;
}

void showImage(const cv::Mat &img, const std::string &title) {
  if (img.empty()) {
    std::cerr << "ERROR: Can't show empty image" << std::endl;
  }
  cv::imshow(title, img);
  cv::waitKey(0);
}
