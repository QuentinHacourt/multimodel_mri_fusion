#ifndef IFUSIONSTRATEGY_H_
#define IFUSIONSTRATEGY_H_

#include <opencv2/opencv.hpp>
#include <vector>

class IFusionStrategy {
  public:
    virtual ~IFusionStrategy() = default;

    virtual cv::Mat fuse(const std::vector<cv::Mat> &images) = 0;
};

#endif
