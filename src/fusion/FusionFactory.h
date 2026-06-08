#ifndef FUSIONFACTORY_H_
#define FUSIONFACTORY_H_
#include "IFusionStrategy.h"
#include "LaplacianPyramidStrategy.h"
#include "PrincipalComponentsStrategy.h"
#include "WaveletBasedStrategy.h"
#include "WeightedAverageStrategy.h"
#include <memory>

class FusionFactory {
  public:
    enum class Type {
        WeightedAverage,
        PrincipalComponents,
        Wavelet,
        Laplacian
    };

    static std::unique_ptr<IFusionStrategy>
    create(Type type, const std::vector<float> &params = {}) {
        switch (type) {
        case Type::WeightedAverage:
            return std::make_unique<WeightedAverageStrategy>(params);
        case Type::PrincipalComponents:
            return std::make_unique<PrincipalComponentsStrategy>();
        case Type::Wavelet:
            return std::make_unique<WaveletBasedStrategy>();
        case Type::Laplacian:
            return std::make_unique<LaplacianPyramidStrategy>();
        default:
            return nullptr;
        }
    }
};

#endif // FUSIONFACTORY_H_
