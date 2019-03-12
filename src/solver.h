// Author: Tucker Haydon

#pragma once

#include <vector>

#include "path_constraint.h"

namespace mediation_layer {
  /* Class for solving piecewise polynomial fitting & minimization problems.
   *
   * Given a polynomial of the following form:
   *   p_1(t) = c_10 (1/0! t^0) + c_11 (1/1! t^1) + c_12 (1/2! t^2) + c_13 (1/3! t^3) + ...
   *   p_2(t) = c_20 (1/0! t^0) + c_21 (1/1! t^1) + c_22 (1/2! t^2) + c_23 (1/3! t^3) + ...
   *   p_3(t) = c_30 (1/0! t^0) + c_31 (1/1! t^1) + c_32 (1/2! t^2) + c_33 (1/3! t^3) + ...
   *   ...
   *
   * finds the minimum of the following cost function:
   *   c^T H c
   *
   * subject to continuity and path constraints.
   *
   * Notes: 
   * 1) polynomial_order must be 3 or more orders greater than derivative_order
   * 2) continuity_order must be less than polynomial_order.
   */
  template <size_t dimension, size_t polynomial_order, size_t derivative_order, size_t continuity_order>
  class PolynomialSolver {
    public:
      struct Options;
      struct Solution;
  
      PolynomialSolver(const Options& options = Options())
        : options_(options);
  
      Solution Run(
          const std::vector<double>& times,
          const std::vector<EqualityBound>& equality_bounds,
          const std::vector<UpperBound>& upper_bounds,
          const std::vector<LowerBound>& lower_bounds);
  
    private:
      Options options_;
  };
  
  struct PolynomialSolver::Options {
    bool multithread = false;
    size_t num_threads = 0;
  
    Options() {}
  };
  
  struct PolynomialSolver::Solution {
  
    Solution() {}
  };
}
