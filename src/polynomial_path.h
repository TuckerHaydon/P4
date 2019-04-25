// Author: Tucker Haydon

#pragma once

#include <vector>
#include <Eigen/Dense>

namespace p4 {
  // Encapsulates information about the piecewise polynomial solution.
  struct PolynomialPath {
    // Vector containing the polynomial coefficient solutions for each
    // dimension. The rows of the vector indicate the dimension, each column
    // of the matrix contains the coefficients for each node, and the row
    // number specifies the coefficient index.
    std::vector<Eigen::MatrixXd> coefficients;

    // Returns the optimal cost of the optimization problem J = 0.5 * x'.P.x
    double optimal_cost;
  
    PolynomialPath(const std::vector<Eigen::MatrixXd>& coefficients_ = {})
      : coefficients(coefficients_) {}
  };
}
