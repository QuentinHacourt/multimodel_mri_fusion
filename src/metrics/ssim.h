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
    double variance(cv::Mat img) {
        const int rows = img.rows;
        const int cols = img.cols;
        const double denominator = (rows * cols) - 1;
        const double mimg = cv::mean(img).val[0];

        double numerator = 0;

        for (int i; i < rows; i++) {
            for (int j; j < cols; j++) {
                numerator += pow((img.at<double>(i, j) - mimg), 2);
            }
        }

        return numerator / denominator;
    }

    double covariance(cv::Mat img1, cv::Mat img2) {
        const int rows = img1.rows;
        const int cols = img1.cols;
        const double denominator = (rows * cols) - 1;
        const double mimg1 = cv::mean(img1).val[0];
        const double mimg2 = cv::mean(img2).val[0];

        double numerator = 0;

        for (int i; i < rows; i++) {
            for (int j; j < cols; j++) {
                numerator += (img1.at<double>(i, j) - mimg1) *
                             (img2.at<double>(i, j) - mimg2);
            }
        }

        return numerator / denominator;
    }

    double localQualityIndex(cv::Mat img1, cv::Mat img2) {
        const double m1 = cv::mean(img1).val[0];
        const double m2 = cv::mean(img2).val[0];
        const double numerator = 4 * covariance(img1, img2) * m1 * m2;
        const double denominator =
            (pow(m1, 2) + pow(m2, 2)) *
            (pow(variance(img1), 2) * pow(variance(img2), 2));

        return numerator / denominator;
    }

    double QualityIndex(const cv::Mat img1, const cv::Mat img2,
                        const int offset = 10) {
        const int rows = img1.rows;
        const int cols = img1.cols;

        int counter = 0;
        double sum = 0;

        for (int i = offset; i < rows - offset; i++) {
            for (int j = offset; i < cols - offset; i++) {
                counter++;

                const cv::Mat a = img1(cv::Range(i - offset, i + offset),
                                       cv::Range(j - offset, j + offset));

                const cv::Mat b = img2(cv::Range(i - offset, i + offset),
                                       cv::Range(j - offset, j + offset));

                sum += localQualityIndex(a, b);
            }
        }

        return sum / counter;
    }

    std::vector<double> lambdas(std::vector<cv::Mat> imgs) {
        std::vector<double> result{};

        // makes it slightly faster
        result.reserve(imgs.size());

        for (int i = 0; i < imgs.size(); i++) {
            result.push_back(variance(imgs[i]));
        }

        return result;
    }

    double sumVector(std::vector<double> v) {
        double sum = 0;

        for (double num : v)
            sum += num;

        return sum;
    }

    std::vector<double> lambdaIs(std::vector<cv::Mat> imgs) {
        std::vector<double> lmbds = lambdas(imgs);
        double sumlmbds = sumVector(lmbds);
        std::vector<double> lmbdis{};

        // makes it slightly faster
        lmbdis.reserve(imgs.size());

        // very funny joke
        for (int c = 0; c < imgs.size(); c++)
            lmbdis.push_back(lmbds[c] / sumlmbds);

        return lmbdis;
    }

    double C(std::vector<double> lmbds) {
        double max = -std::numeric_limits<double>::infinity();

        for (double l : lmbds)
            if (l > max)
                max = l;

        return max;
    }

    double c(double C, double sumCs) { return C / sumCs; }

  public:
    double metric(std::vector<cv::Mat> imgs, cv::Mat fused) {
        double res = 0;
        std::vector<double> rs{};
        const int rows = imgs[1].rows;
        const int cols = imgs[1].cols;
        int offset = 20;

        std::vector<double> Ceeees{};

        for (int i = offset; i < rows - offset; i++) {
            for (int j = offset; i < cols - offset; i++) {
                double r = 0;
                const cv::Mat wf = fused(cv::Range(i - offset, i + offset),
                                         cv::Range(j - offset, j + offset));

                std::vector<cv::Mat> ws{};

                for (cv::Mat img : imgs) {
                    const cv::Mat w = img(cv::Range(i - offset, i + offset),
                                          cv::Range(j - offset, j + offset));

                    ws.push_back(w);
                }

                std::vector<double> lmbdIs = lambdaIs(ws);

                Ceeees.push_back(C(lambdas(ws)));

                for (int i = 0; i < imgs.size(); i++) {
                    r += lmbdIs[i] * localQualityIndex(ws[i], wf);
                }

                rs.push_back(r);
            }
        }

        double sumCeeees = sumVector(Ceeees);

        for (int i = 0; i < rs.size(); i++)
            res += (Ceeees[i] / sumCeeees) * rs[i];

        return res;
    }
};

#endif // SSIM_H_
