#ifndef PRINCIPALCOMPONENTSSTRATEGY_H_
#define PRINCIPALCOMPONENTSSTRATEGY_H_
#include "IFusionStrategy.h"
#include <Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Eigenvalues>
#include <numeric>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/mat.hpp>
#include <vector>

class PrincipalComponentsStrategy : public IFusionStrategy {
  private:
    uchar approximate(Eigen::VectorXd x, Eigen::VectorXd m_x,
                      Eigen::MatrixXd C_x) {

        int n = x.size();

        /* Eigen::MatrixXd mat(n, n); */
        /* for (int i = 0; i < n; i++) */
        /*     for (int j = 0; j < n; j++) */
        /*         mat(i, j) = static_cast<double>(C_x[i][j]); */

        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(C_x);
        Eigen::VectorXd eigenvalues = solver.eigenvalues();
        Eigen::MatrixXd eigenvectors = solver.eigenvectors();

        /* for (int i = 0; i < n; i++) */
        /*     std::cout << eigenvalues[i] << std::endl; */

        Eigen::MatrixXd A = sorted(eigenvalues, eigenvectors);

        /* Eigen::VectorXd y = A * (x - m_x); */

        Eigen::VectorXd A_k = A.row(0);

        double fused = A_k.dot(x - m_x) + m_x.mean();

        return static_cast<uchar>(std::clamp(fused, 0.0, 255.0));
    }

    Eigen::MatrixXd sorted(Eigen::VectorXd eigenvalues,
                           Eigen::MatrixXd eigenvectors) {
        int n = eigenvalues.size();

        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](int a, int b) {
            return eigenvalues[a] > eigenvalues[b];
        });

        Eigen::MatrixXd sorted(eigenvectors.rows(), n);
        for (int i = 0; i < n; i++)
            sorted.row(i) = eigenvectors.col(indices[i]).transpose();

        return sorted;
    }

  public:
    explicit PrincipalComponentsStrategy() {}

    cv::Mat fuse(const std::vector<cv::Mat> &images) override {
        int64 n = images.size();
        int64 rows = images[0].rows;
        int64 cols = images[0].cols;
        int64 pixels = rows * cols;
        cv::Mat res = cv::Mat::zeros(images[0].size(), images[0].type());

        // expected value for each image
        Eigen::VectorXd m_x = Eigen::VectorXd::Zero(n);

        // covariance matrix
        Eigen::MatrixXd C_x = Eigen::MatrixXd::Zero(n, n);

        for (int64 i = 0; i < rows; i++) {
            for (int64 j = 0; j < cols; j++) {
                for (int64 k = 0; k < n; k++) {
                    m_x[k] += images[k].at<uchar>(i, j);
                }
            }
        }

        // expectance vector
        for (int64 k = 0; k < n; k++)
            m_x[k] = m_x[k] / pixels;

        for (int64 row = 0; row < rows; row++) {
            for (int64 col = 0; col < cols; col++) {

                Eigen::VectorXd x = Eigen::VectorXd::Zero(n);

                for (int64 i = 0; i < n; i++)
                    x[i] = images[i].at<uchar>(row, col);

                C_x += (x - m_x) * (x - m_x).transpose();

                /* res.at<uchar>(i, j) = approximate(x, m_x, C_x); */
            }
        }

        C_x /= (pixels - 1);

        for (int64 row = 0; row < rows; row++) {
            for (int64 col = 0; col < cols; col++) {

                Eigen::VectorXd x = Eigen::VectorXd::Zero(n);

                for (int64 i = 0; i < n; i++)
                    x[i] = images[i].at<uchar>(row, col);

                res.at<uchar>(row, col) = approximate(x, m_x, C_x);
            }
        }
        return res;
    }
};

#endif // PRINCIPALCOMPONENTSSTRATEGY_H_
