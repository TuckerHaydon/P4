// Author: Tucker Haydon

#pragma once

#include <vector>
#include <Eigen/Dense>

namespace mediation_layer {
  // Encapsulates information about the piecewise polynomial solution.
  struct PolynomialPath {
    // Vector containing the polynomial coefficient solutions for each
    // dimension. The rows of the vector indicate the dimension, each column
    // of the matrix contains the coefficients for each node, and the row
    // number specifies the coefficient index.
    std::vector<Eigen::MatrixXd> coefficients;
  
    PolynomialPath(const std::vector<Eigen::MatrixXd>& coefficients_ = {})
      : coefficients(coefficients_) {}
  };
}
