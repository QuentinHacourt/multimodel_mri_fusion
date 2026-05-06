#ifndef IO_H_
#define IO_H_

#include <SimpleITK.h>
#include <opencv2/opencv.hpp>
#include <string>

cv::Mat loadImage(const std::string &imagePath);
void showImage(const cv::Mat &img, const std::string &title);

#endif // IO_H_
