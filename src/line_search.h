// Author: Tucker Haydon

#pragma once

#include "polynomial_solver.h"
#include "gradient.h"

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
        const std::vector<double>& initial_times,
        const std::shared_ptr<PolynomialSolver>& solver,
        const PolynomialSolver::Solution& initial_solver_solution,
        const Gradient::Solution& gradient_solution) const;

    private:
      Options options_;
  };
}
