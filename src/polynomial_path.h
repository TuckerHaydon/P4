// Author: Tucker Haydon

#pragma once

#include <vector>
#include <Eigen/Dense>
#include <osqp.h>

namespace p4 {
  // Encapsulates information about the piecewise polynomial solution.
  struct PolynomialPath {
    // Vector containing the polynomial coefficient solutions for each
    // dimension. The rows of the vector indicate the dimension, each column
    // of the matrix contains the coefficients for each node, and the row
    // number specifies the coefficient index.
    std::vector<Eigen::MatrixXd> coefficients;

    // OSQP Output information. Contains information about whether a solution
    // was found, how long it took to find the solution, the optimal cost, etc.
    //   osqp_info.obj_val: the optimal cost of the optimization problem J = 0.5 * x'.P.x
    // Reference: https://osqp.org/docs/interfaces/cc++#_CPPv48OSQPInfo
    OSQPInfo osqp_info;
  
    PolynomialPath(const std::vector<Eigen::MatrixXd>& coefficients_ = {})
      : coefficients(coefficients_) {}
  };
}
