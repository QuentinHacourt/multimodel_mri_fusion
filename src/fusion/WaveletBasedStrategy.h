#ifndef WAVELETBASEDSTRATEGY_H_
#define WAVELETBASEDSTRATEGY_H_

#include "IFusionStrategy.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <opencv2/opencv.hpp>
#include <vector>

class WaveletBasedStrategy : public IFusionStrategy {
  private:
    const int NUM_LEVELS = 3;

    const std::vector<double> L = {1.0 / sqrt(2.0), 1.0 / sqrt(2.0)};
    const std::vector<double> H = {1.0 / sqrt(2.0), -1.0 / sqrt(2.0)};

    struct SubbandLevel {
        cv::Mat LH, HL, HH;
    };

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

        cv::Mat res = cv::Mat::zeros(rows / 2, cols, img.type());

        for (int k = 0; k < rows / 2; k++) {
            for (int j = 0; j < cols; j++) {
                double val = 0;
                for (int f = 0; f < flen; f++) {
                    int row = mirrorIndex(2 * k + f, rows);
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

        cv::Mat res = cv::Mat::zeros(rows, cols / 2, img.type());

        for (int i = 0; i < rows; i++) {
            for (int k = 0; k < cols / 2; k++) {
                double val = 0;
                for (int f = 0; f < flen; f++) {
                    int col = mirrorIndex(2 * k + f, cols);
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

        cv::Mat res = cv::Mat::zeros(rows, targetCols, img.type());

        for (int i = 0; i < rows; i++) {
            for (int k = 0; k < img.cols; k++) {
                double val = img.at<double>(i, k);
                for (int f = 0; f < flen; f++) {
                    int col = 2 * k + f;
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

        cv::Mat res = cv::Mat::zeros(targetRows, cols, img.type());

        for (int k = 0; k < img.rows; k++) {
            for (int j = 0; j < cols; j++) {
                double val = img.at<double>(k, j);
                for (int f = 0; f < flen; f++) {
                    int row = 2 * k + f;
                    if (row >= 0 && row < targetRows)
                        res.at<double>(row, j) += filter[f] * val;
                }
            }
        }
        return res;
    }

    void decomposeLevel(cv::Mat img, cv::Mat &LL, cv::Mat &LH, cv::Mat &HL,
                        cv::Mat &HH) {
        cv::Mat rowL = filterRows(img, L);
        cv::Mat rowH = filterRows(img, H);
        LL = filterCols(rowL, L);
        LH = filterCols(rowL, H);
        HL = filterCols(rowH, L);
        HH = filterCols(rowH, H);
    }

    cv::Mat reconstructLevel(cv::Mat LL, cv::Mat LH, cv::Mat HL, cv::Mat HH) {
        const int rows = LL.rows * 2;
        const int cols = LL.cols * 2;

        cv::Mat upLL = upsampleFilterCols(LL, L, rows);
        cv::Mat upLH = upsampleFilterCols(LH, H, rows);
        cv::Mat upHL = upsampleFilterCols(HL, L, rows);
        cv::Mat upHH = upsampleFilterCols(HH, H, rows);

        cv::Mat rowL = upLL + upLH;
        cv::Mat rowH = upHL + upHH;

        cv::Mat resL = upsampleFilterRows(rowL, L, cols);
        cv::Mat resH = upsampleFilterRows(rowH, H, cols);

        return resL + resH;
    }

    cv::Mat multiLevelDecompose(cv::Mat img,
                                std::vector<SubbandLevel> &levels) {
        levels.clear();
        cv::Mat current = img;

        for (int l = 0; l < NUM_LEVELS; l++) {
            cv::Mat LL, LH, HL, HH;
            decomposeLevel(current, LL, LH, HL, HH);

            SubbandLevel sl;
            sl.LH = LH;
            sl.HL = HL;
            sl.HH = HH;
            levels.push_back(sl);

            current = LL;
        }

        return current;
    }

    cv::Mat multiLevelReconstruct(cv::Mat deepLL,
                                  const std::vector<SubbandLevel> &levels) {
        cv::Mat current = deepLL;

        for (int l = NUM_LEVELS - 1; l >= 0; l--) {
            current = reconstructLevel(current, levels[l].LH, levels[l].HL,
                                       levels[l].HH);
        }

        return current;
    }

    cv::Mat fuseLL(const std::vector<cv::Mat> &LLbands) {
        cv::Mat res = cv::Mat::zeros(LLbands[0].size(), LLbands[0].type());
        const double weight = 1.0 / LLbands.size();
        for (const cv::Mat &band : LLbands)
            res += weight * band;
        return res;
    }

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

  public:
    explicit WaveletBasedStrategy() {}

    cv::Mat fuse(const std::vector<cv::Mat> &images) override {
        if (images.empty())
            return cv::Mat();

        std::vector<cv::Mat> converted;
        for (const cv::Mat &img : images) {
            cv::Mat conv;
            img.convertTo(conv, CV_64F);
            converted.push_back(conv);
        }

        std::vector<cv::Mat> deepLLs;
        std::vector<std::vector<SubbandLevel>> allLevels(converted.size());

        for (int i = 0; i < converted.size(); i++) {
            cv::Mat deepLL = multiLevelDecompose(converted[i], allLevels[i]);
            deepLLs.push_back(deepLL);
        }

        cv::Mat fusedLL = fuseLL(deepLLs);

        std::vector<SubbandLevel> fusedLevels(NUM_LEVELS);

        for (int l = 0; l < NUM_LEVELS; l++) {
            std::vector<cv::Mat> LHbands, HLbands, HHbands;

            for (int i = 0; i < converted.size(); i++) {
                LHbands.push_back(allLevels[i][l].LH);
                HLbands.push_back(allLevels[i][l].HL);
                HHbands.push_back(allLevels[i][l].HH);
            }

            if (l == 0) {
                fusedLevels[l].LH = fuseLL(LHbands);
                fusedLevels[l].HL = fuseLL(HLbands);
                fusedLevels[l].HH = fuseLL(HHbands);
            } else {
                fusedLevels[l].LH = fuseDetail(LHbands);
                fusedLevels[l].HL = fuseDetail(HLbands);
                fusedLevels[l].HH = fuseDetail(HHbands);
            }
        }

        cv::Mat fusedDouble = multiLevelReconstruct(fusedLL, fusedLevels);

        cv::Mat finalRes;
        fusedDouble.convertTo(finalRes, CV_8U);

        cv::GaussianBlur(finalRes, finalRes, cv::Size(3, 3), 0.5);

        return finalRes;
    }
};

#endif // WAVELETBASEDSTRATEGY_H_
