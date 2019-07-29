// Author: Tucker Haydon

#pragma once

#include "polynomial_solver.h"

namespace p4 {
  class LineSearch {
    public:
      struct Options {
        double alpha_0 = 1.0;
        size_t max_iterations = 10;

        Options() {}
      };

      struct Solution {
        std::vector<double> times;

        Solution() {}
      };

      LineSearch(const Options& options)
        : options_(options) {}

      Solution Run(
          const std::vector<double>& times,
          const std::shared_ptr<PolynomialSolver>& solver);

    private:
      Options options_;
  };
}
