#ifndef WAVELETBASEDSTRATEGY_H_
#define WAVELETBASEDSTRATEGY_H_

#include "IFusionStrategy.h"
#include <cmath>
class WaveletBasedStrategy : public IFusionStrategy {
  private:
    const double SQRT2 = std::sqrt(2.0);

    const std::vector<double> L_analysis = {
        -1.0 / 8.0 * SQRT2, 2.0 / 8.0 * SQRT2, 6.0 / 8.0 * SQRT2,
        2.0 / 8.0 * SQRT2, -1.0 / 8.0 * SQRT2};
    const std::vector<double> H_analysis = {1.0 / 2.0 * SQRT2, -1.0 * SQRT2,
                                            1.0 / 2.0 * SQRT2};

    const std::vector<double> L_synthesis = {1.0 / 2.0 / SQRT2, 1.0 / SQRT2,
                                             1.0 / 2.0 / SQRT2};
    const std::vector<double> H_synthesis = {
        1.0 / 8.0 / SQRT2, 2.0 / 8.0 / SQRT2, -6.0 / 8.0 / SQRT2,
        2.0 / 8.0 / SQRT2, 1.0 / 8.0 / SQRT2};

    int mirrorIndex(int idx, int size) {
        if (idx < 0)
            return -idx;
        if (idx >= size)
            return 2 * size - idx - 2;
        return idx;
    }

    cv::Mat filterCols(cv::Mat img, const std::vector<double> &filter) {
        const int rows = img.rows;
        const int cols = img.cols;
        const int flen = filter.size();

        const int half = 2;
        const int offset = half - (flen / 2);

        cv::Mat res = cv::Mat::zeros(rows / 2, cols, img.type());

        for (int k = 0; k < rows / 2; k++) {
            for (int j = 0; j < cols; j++) {
                double val = 0;
                for (int f = 0; f < flen; f++) {
                    int row = mirrorIndex(2 * k + f - half + offset, rows);
                    val += filter[f] * img.at<double>(row, j);
                }
                res.at<double>(k, j) = val;
            }
        }
        return res;
    }

    cv::Mat filterRows(cv::Mat img, const std::vector<double> &filter) {
        const int rows = img.rows;
        const int cols = img.cols;
        const int flen = filter.size();

        const int half = 2;
        const int offset = half - (flen / 2);

        cv::Mat res = cv::Mat::zeros(rows, cols / 2, img.type());

        for (int i = 0; i < rows; i++) {
            for (int k = 0; k < cols / 2; k++) {
                double val = 0;
                for (int f = 0; f < flen; f++) {
                    int col = mirrorIndex(2 * k + f - half + offset, cols);
                    val += filter[f] * img.at<double>(i, col);
                }
                res.at<double>(i, k) = val;
            }
        }
        return res;
    }

    cv::Mat upsampleFilterRows(cv::Mat img, const std::vector<double> &filter,
                               int targetCols) {
        const int rows = img.rows;
        const int flen = filter.size();

        const int half = 2;
        const int offset = half - (flen / 2);

        cv::Mat res = cv::Mat::zeros(rows, targetCols, img.type());

        for (int i = 0; i < rows; i++) {
            for (int k = 0; k < img.cols; k++) {
                double val = img.at<double>(i, k);
                for (int f = 0; f < flen; f++) {
                    int col = 2 * k + f - half + offset;
                    if (col >= 0 && col < targetCols)
                        res.at<double>(i, col) += filter[f] * val;
                }
            }
        }
        return res;
    }

    cv::Mat upsampleFilterCols(cv::Mat img, const std::vector<double> &filter,
                               int targetRows) {
        const int cols = img.cols;
        const int flen = filter.size();

        const int half = 2;
        const int offset = half - (flen / 2);

        cv::Mat res = cv::Mat::zeros(targetRows, cols, img.type());

        for (int k = 0; k < img.rows; k++) {
            for (int j = 0; j < cols; j++) {
                double val = img.at<double>(k, j);
                for (int f = 0; f < flen; f++) {
                    int row = 2 * k + f - half + offset;
                    if (row >= 0 && row < targetRows)
                        res.at<double>(row, j) += filter[f] * val;
                }
            }
        }

        return res;
    }

    // decomposition in two dimensions
    void decompose(cv::Mat img, cv::Mat &LL, cv::Mat &LH, cv::Mat &HL,
                   cv::Mat &HH) {
        cv::Mat rowL = filterRows(img, L_analysis);
        cv::Mat rowH = filterRows(img, H_analysis);

        // approximation
        LL = filterCols(rowL, L_analysis);

        // the rest are details
        LH = filterCols(rowL, H_analysis);
        HL = filterCols(rowH, L_analysis);
        HH = filterCols(rowH, H_analysis);
    }

    // WA for LL
    cv::Mat fuseLL(const std::vector<cv::Mat> &LLbands) {
        cv::Mat res = cv::Mat::zeros(LLbands[0].size(), LLbands[0].type());
        const double weight = 1.0 / LLbands.size();

        for (const cv::Mat &band : LLbands)
            res += weight * band;

        return res;
    }

    // choose-max for details
    cv::Mat fuseDetail(const std::vector<cv::Mat> &bands) {
        const int rows = bands[0].rows;
        const int cols = bands[0].cols;
        cv::Mat res = cv::Mat::zeros(rows, cols, bands[0].type());

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double maxVal = -std::numeric_limits<double>::infinity();
                double chosen = 0;

                for (const cv::Mat &band : bands) {
                    double val = band.at<double>(i, j);
                    if (std::abs(val) > maxVal) {
                        maxVal = std::abs(val);
                        chosen = val;
                    }
                }
                res.at<double>(i, j) = chosen;
            }
        }
        return res;
    }

    cv::Mat reconstruct(cv::Mat LL, cv::Mat LH, cv::Mat HL, cv::Mat HH) {
        const int rows = LL.rows * 2;
        const int cols = LL.cols * 2;

        cv::Mat upLL = upsampleFilterCols(LL, L_synthesis, rows);
        cv::Mat upLH = upsampleFilterCols(LH, H_synthesis, rows);
        cv::Mat upHL = upsampleFilterCols(HL, L_synthesis, rows);
        cv::Mat upHH = upsampleFilterCols(HH, H_synthesis, rows);

        cv::Mat rowL = upLL + upLH;
        cv::Mat rowH = upHL + upHH;

        cv::Mat resL = upsampleFilterRows(rowL, L_synthesis, cols);
        cv::Mat resH = upsampleFilterRows(rowH, H_synthesis, cols);

        return resL + resH;
    }

  public:
    explicit WaveletBasedStrategy() {}

    cv::Mat fuse(const std::vector<cv::Mat> &images) override {

        cv::Mat test;
        images[0].convertTo(test, CV_64F);

        // decompose and immediately reconstruct WITHOUT fusion
        cv::Mat LL, LH, HL, HH;
        decompose(test, LL, LH, HL, HH);
        cv::Mat reconstructed = reconstruct(LL, LH, HL, HH);

        // compute difference
        cv::Mat diff;
        cv::absdiff(test, reconstructed, diff);
        std::cout << "Max reconstruction error: "
                  << *std::max_element(diff.begin<double>(), diff.end<double>())
                  << std::endl;

        // real code
        if (images.empty())
            return cv::Mat();

        std::vector<cv::Mat> converted;
        for (const cv::Mat &img : images) {
            cv::Mat conv;
            img.convertTo(conv, CV_64F);
            converted.push_back(conv);
        }

        std::vector<cv::Mat> LLbands, LHbands, HLbands, HHbands;

        for (const cv::Mat &img : converted) {
            cv::Mat LL, LH, HL, HH;
            decompose(img, LL, LH, HL, HH);
            LLbands.push_back(LL);
            LHbands.push_back(LH);
            HLbands.push_back(HL);
            HHbands.push_back(HH);
        }

        cv::Mat fusedLL = fuseLL(LLbands);
        cv::Mat fusedLH = fuseDetail(LHbands);
        cv::Mat fusedHL = fuseDetail(HLbands);
        cv::Mat fusedHH = fuseDetail(HHbands);

        cv::Mat fusedDouble = reconstruct(fusedLL, fusedLH, fusedHL, fusedHH);

        cv::Mat finalRes;
        fusedDouble.convertTo(finalRes, CV_8U);

        return finalRes;
    }
};

#endif // WAVELETBASEDSTRATEGY_H_
