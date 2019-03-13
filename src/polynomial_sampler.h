// Author: Tucker Haydon

#pragma once

#include <Eigen/Dense>

#include <vector>

#include "polynomial_path.h"

namespace mediation_layer {
  class PolynomialSampler {
    public:
      struct Options {
        double frequency = 20;

        Options(){}
      };

      PolynomialSampler(const Options& options = Options())
        : options_(options) {}

      Eigen::MatrixXd Run(const std::vector<double>& times, const PolynomialPath& path);

    private:
      Options options_;

  };
}
