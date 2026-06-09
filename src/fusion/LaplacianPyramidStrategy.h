#ifndef LAPLACIANPYRAMIDSTRATEGY_H_
#define LAPLACIANPYRAMIDSTRATEGY_H_

#include "IFusionStrategy.h"
#include <opencv2/core/mat.hpp>
class LaplacianPyramidStrategy : public IFusionStrategy {
  private:
    const int NUM_LEVELS = 3;

    const std::vector<double> GAUSSIAN = {1.0 / 16, 4.0 / 16, 6.0 / 16,
                                          4.0 / 16, 1.0 / 16};

    int mirrorIndex(int idx, int size) {
        if (idx < 0)
            return -idx;
        if (idx >= size)
            return 2 * size - idx - 2;
        return idx;
    }

    cv::Mat filterCols(cv::Mat img) {
        const int rows = img.rows;
        const int cols = img.cols;
        const int flen = GAUSSIAN.size();

        cv::Mat res = cv::Mat::zeros(rows / 2, cols, img.type());

        for (int k = 0; k < rows / 2; k++) {
            for (int j = 0; j < cols; j++) {
                double val = 0;
                for (int f = 0; f < flen; f++) {
                    int row = mirrorIndex(2 * k + f - 2, rows);
                    val += GAUSSIAN[f] * img.at<double>(row, j);
                }
                res.at<double>(k, j) = val;
            }
        }
        return res;
    }

    cv::Mat filterRows(cv::Mat img) {
        const int rows = img.rows;
        const int cols = img.cols;
        const int flen = GAUSSIAN.size();

        cv::Mat res = cv::Mat::zeros(rows, cols / 2, img.type());

        for (int i = 0; i < rows; i++) {
            for (int k = 0; k < cols / 2; k++) {
                double val = 0;
                for (int f = 0; f < flen; f++) {
                    int col = mirrorIndex(2 * k + f - 2, cols);
                    val += GAUSSIAN[f] * img.at<double>(i, col);
                }
                res.at<double>(i, k) = val;
            }
        }
        return res;
    }

    cv::Mat upsampleRows(cv::Mat img, int targetCols) {
        const int rows = img.rows;
        const int flen = GAUSSIAN.size();

        cv::Mat res = cv::Mat::zeros(rows, targetCols, img.type());

        for (int i = 0; i < rows; i++) {
            for (int k = 0; k < img.cols; k++) {
                double val = img.at<double>(i, k);
                for (int f = 0; f < flen; f++) {
                    int col = 2 * k + f;
                    if (col >= 0 && col < targetCols)
                        res.at<double>(i, col) += GAUSSIAN[f] * val;
                }
            }
        }
        return res;
    }

    cv::Mat upsampleCols(cv::Mat img, int targetRows) {
        const int cols = img.cols;
        const int flen = GAUSSIAN.size();

        cv::Mat res = cv::Mat::zeros(targetRows, cols, img.type());

        for (int k = 0; k < img.rows; k++) {
            for (int j = 0; j < cols; j++) {
                double val = img.at<double>(k, j);
                for (int f = 0; f < flen; f++) {
                    int row = 2 * k + f;
                    if (row >= 0 && row < targetRows)
                        res.at<double>(row, j) += GAUSSIAN[f] * val;
                }
            }
        }
        return res;
    }

    void decomposeLevel(cv::Mat img, cv::Mat &G) {
        // filter rows
        cv::Mat row = filterRows(img);
        // filter cols
        G = filterCols(row);
    }

    cv::Mat reconstructLevel(cv::Mat G) {
        const int rows = G.rows * 2;
        const int cols = G.cols * 2;

        cv::Mat upGCol = upsampleCols(G, rows);

        cv::Mat resG = upsampleRows(upGCol, cols);

        return resG;
    }

    cv::Mat multiLevelDecompose(cv::Mat img,
                                std::vector<cv::Mat> &gaussianPyramid) {
        gaussianPyramid.clear();
        gaussianPyramid.push_back(img);
        cv::Mat current = img;

        for (int l = 0; l < NUM_LEVELS; l++) {
            cv::Mat G;
            decomposeLevel(current, G);
            gaussianPyramid.push_back(G);

            current = G;
        }

        return current;
    }

    cv::Mat
    multiLevelReconstruct(const std::vector<cv::Mat> &laplacianPyramid) {
        cv::Mat current = laplacianPyramid.back();

        for (int l = laplacianPyramid.size() - 2; l >= 0; l--) {
            current = reconstructLevel(current) + laplacianPyramid[l];
        }

        return current;
    }

    cv::Mat weightedAverages(const std::vector<cv::Mat> &Gs) {
        cv::Mat res = cv::Mat::zeros(Gs[0].size(), Gs[0].type());
        std::vector<float> weights = {0.1, 0.3, 0.3, 0.3};
        for (int i = 0; i < Gs.size(); i++)
            res += Gs[i] * weights[i];

        return res;
    }

    cv::Mat chooseMax(const std::vector<cv::Mat> &Ls) {
        const int rows = Ls[0].rows;
        const int cols = Ls[0].cols;
        cv::Mat res = cv::Mat::zeros(rows, cols, Ls[0].type());

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                double maxVal = -std::numeric_limits<double>::infinity();
                double chosen = 0;
                for (const cv::Mat &band : Ls) {
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

    std::vector<cv::Mat> GsToLs(std::vector<cv::Mat> Gs) {
        std::vector<cv::Mat> Ls{};
        for (int i = 0; i < Gs.size(); i++) {
            if (i == Gs.size() - 1) {
                Ls.push_back(Gs[i]);
            } else {
                Ls.push_back(Gs[i] - reconstructLevel(Gs[i + 1]));
            }
        }

        return Ls;
    }

  public:
    explicit LaplacianPyramidStrategy() {}

    cv::Mat fuse(const std::vector<cv::Mat> &images) override {
        if (images.empty())
            return cv::Mat();

        std::vector<cv::Mat> converted;
        for (const cv::Mat &img : images) {
            cv::Mat conv;
            img.convertTo(conv, CV_64F);
            converted.push_back(conv);
        }

        std::vector<std::vector<cv::Mat>> allLaplacians(converted.size());

        for (int i = 0; i < converted.size(); i++) {
            std::vector<cv::Mat> gaussianPyramid;
            multiLevelDecompose(converted[i], gaussianPyramid);
            allLaplacians[i] = GsToLs(gaussianPyramid);
        }

        int pyramidSize = allLaplacians[0].size();
        std::vector<cv::Mat> fusedPyramid(pyramidSize);

        for (int l = 0; l < pyramidSize; l++) {
            std::vector<cv::Mat> levelBands;

            for (int i = 0; i < converted.size(); i++) {
                levelBands.push_back(allLaplacians[i][l]);
            }

            if (l == pyramidSize - 1) {
                fusedPyramid[l] = weightedAverages(levelBands);
            } else {
                fusedPyramid[l] = chooseMax(levelBands);
            }
        }

        // reconstruct
        cv::Mat fusedDouble = multiLevelReconstruct(fusedPyramid);

        cv::Mat finalRes;
        fusedDouble.convertTo(finalRes, CV_8U);
        return finalRes;
    }
};

#endif // LAPLACIANPYRAMIDSTRATEGY_H_
