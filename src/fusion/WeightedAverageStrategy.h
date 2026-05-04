#ifndef WEIGHTEDAVERAGESTRATEGY_H_
#define WEIGHTEDAVERAGESTRATEGY_H_

#include "IFusionStrategy.h"
class WeightedAverageStrategy : public IFusionStrategy {
  private:
    std::vector<float> weights;

    std::vector<float> normalize() {
        float sum = 0;
        for (float weight : weights)
            sum += weight;

        for (float &weight : weights)
            weight /= sum;

        return weights;
    }

  public:
    explicit WeightedAverageStrategy(const std::vector<float> &weights)
        : weights(weights) {}

    cv::Mat fuse(const std::vector<cv::Mat> &images) override {
        cv::Mat res = cv::Mat::zeros(images[0].size(), images[0].type());

        normalize();

        for (int i = 0; i < images.size(); i++)
            res += images[i] * weights[i];

        return res;
    }
};

#endif // WEIGHTEDAVERAGESTRATEGY_H_
