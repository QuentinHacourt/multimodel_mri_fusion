#ifndef FUSIONFACTORY_H_
#define FUSIONFACTORY_H_
#include "IFusionStrategy.h"
#include "WeightedAverageStrategy.h"
#include <memory>
#include <string>

class FusionFactory {
  public:
    enum class Type { WeightedAverage };

    static std::unique_ptr<IFusionStrategy>
    create(Type type, const std::vector<float> &params = {}) {
        switch (type) {
        case Type::WeightedAverage:
            return std::make_unique<WeightedAverageStrategy>(params);
        default:
            return nullptr;
        }
    }
};

#endif // FUSIONFACTORY_H_
