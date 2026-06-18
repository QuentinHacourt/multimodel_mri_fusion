#ifndef SSIM_H_
#define SSIM_H_

#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vector>

class StructuralSimilarityIndexMeasure {
  private:
    double variance(const cv::Mat &img) {
        cv::Mat imgDouble;
        img.convertTo(imgDouble, CV_64F);

        const int rows = imgDouble.rows;
        const int cols = imgDouble.cols;
        const double denominator = (rows * cols) - 1;

        if (denominator <= 0)
            return 0.0;

        const double mimg = cv::mean(imgDouble).val[0];
        double numerator = 0;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                numerator += pow((imgDouble.at<double>(i, j) - mimg), 2);
            }
        }

        return numerator / denominator;
    }

    double covariance(const cv::Mat &img1, const cv::Mat &img2) {
        cv::Mat img1Double, img2Double;
        img1.convertTo(img1Double, CV_64F);
        img2.convertTo(img2Double, CV_64F);

        const int rows = img1Double.rows;
        const int cols = img1Double.cols;
        const double denominator = (rows * cols) - 1;
        const double mimg1 = cv::mean(img1Double).val[0];
        const double mimg2 = cv::mean(img2Double).val[0];

        if (denominator <= 0)
            return 0.0;

        double numerator = 0;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                numerator += (img1Double.at<double>(i, j) - mimg1) *
                             (img2Double.at<double>(i, j) - mimg2);
            }
        }

        return numerator / denominator;
    }

    double localQualityIndex(const cv::Mat &w1, const cv::Mat &w2, double var1,
                             double var2) {
        const double m1 = cv::mean(w1).val[0];
        const double m2 = cv::mean(w2).val[0];
        const double covar = covariance(w1, w2);

        const double numerator = 4 * covar * m1 * m2;
        const double denominator = (pow(m1, 2) + pow(m2, 2)) * (var1 + var2);

        if (denominator == 0.0)
            return (m1 == m2 && var1 == var2) ? 1.0 : 0.0;

        return numerator / denominator;
    }

    std::vector<double> lambdas(std::vector<cv::Mat> &imgs) {
        std::vector<double> result{};

        for (size_t i = 0; i < imgs.size(); i++) {
            result.push_back(variance(imgs[i]));
        }

        return result;
    }

    double sumVector(std::vector<double> &v) {
        double sum = 0;

        for (double num : v)
            sum += num;

        return sum;
    }

    std::vector<double> lambdaIs(std::vector<cv::Mat> &imgs) {
        std::vector<double> lmbds = lambdas(imgs);
        double sumlmbds = sumVector(lmbds);
        std::vector<double> lmbdis{};

        for (size_t i = 0; i < imgs.size(); i++) {
            lmbdis.push_back(sumlmbds == 0.0 ? 1.0 / imgs.size()
                                             : lmbds[i] / sumlmbds);
        }

        return lmbdis;
    }

    double C(std::vector<double> &lmbds) {
        double max = -std::numeric_limits<double>::infinity();

        for (double l : lmbds)
            if (l > max)
                max = l;

        return max;
    }

    double c(double C, double sumCs) { return C / sumCs; }

  public:
    double metric(std::vector<cv::Mat> &imgs, cv::Mat &fused) {
        if (imgs.empty() || fused.empty())
            return 0.0;

        double res = 0;
        std::vector<double> rs{};
        const int rows = imgs[0].rows;
        const int cols = imgs[0].cols;
        int offset = 20;

        std::vector<double> Ceeees{};

        for (int i = offset; i < rows - offset; i++) {
            for (int j = offset; j < cols - offset; j++) {
                double r = 0;
                const cv::Mat wf = fused(cv::Range(i - offset, i + offset),
                                         cv::Range(j - offset, j + offset));

                std::vector<cv::Mat> ws{};
                std::vector<double> local_variances{};
                double sum_variances = 0.0;
                double max_variance = -std::numeric_limits<double>::infinity();

                for (const cv::Mat &img : imgs) {
                    cv::Mat w = img(cv::Range(i - offset, i + offset),
                                    cv::Range(j - offset, j + offset));
                    ws.push_back(w);

                    double var = variance(w);
                    local_variances.push_back(var);
                    sum_variances += var;
                    if (var > max_variance) {
                        max_variance = var;
                    }
                }

                if (max_variance < 0)
                    max_variance = 0.0;

                std::vector<double> lmbdIs{};
                for (double var : local_variances) {
                    lmbdIs.push_back(sum_variances == 0.0
                                         ? 1.0 / imgs.size()
                                         : var / sum_variances);
                }

                Ceeees.push_back(max_variance);

                double var_fused = variance(wf);
                for (size_t k = 0; k < imgs.size(); k++) {
                    r += lmbdIs[k] * localQualityIndex(ws[k], wf,
                                                       local_variances[k],
                                                       var_fused);
                }

                rs.push_back(r);
            }
        }

        double sumCeeees = sumVector(Ceeees);

        for (size_t i = 0; i < rs.size(); i++) {
            double weight = (sumCeeees == 0.0) ? (1.0 / rs.size())
                                               : (Ceeees[i] / sumCeeees);
            res += weight * rs[i];
        }

        return res;
    }
};

#endif // SSIM_H_
