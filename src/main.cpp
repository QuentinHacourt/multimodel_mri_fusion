#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
  // Create a blank 400x400 image (Blue background)
  cv::Mat image(400, 400, CV_8UC3, cv::Scalar(255, 0, 0));

  cv::putText(image, "OpenCV + Arch + Doom", cv::Point(50, 200),
              cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 255), 2);

  cv::imshow("Success", image);
  cv::waitKey(0);
  return 0;
}
