// Author: Tucker Haydon

#pragma once

#include "polynomial_solver.h"
#include "gradient.h"
#include "line_search.h"

#include <vector>

namespace p4 {
  class GradientDescent {
    public:
      struct Options {
        PolynomialSolver::Options solver_options;
        LineSearch::Options line_search_options;
        Gradient::Options gradient_options;

        // Maximum number of gradient descent iterations
        size_t max_iterations = 10;
        // Terminate gradient descent if solution cost is less than or equal to
        // this threshhold
        double cost_threshhold = 0.1;

        Options() {}
      };

      struct Solution {
        PolynomialSolver::Solution solver_solution;

        Solution() {}
      };

      GradientDescent(const Options& options)
        : options_(options) {}

      Solution Run(const std::vector<double>& initial_times);

    private:
      Options options_;

  };
}
