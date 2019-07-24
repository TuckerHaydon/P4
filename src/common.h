// Author: Tucker Haydon

#pragma once

#include <Eigen/Dense>
#include <cstdlib>

namespace p4 {
  // Computes n!
  size_t Factorial(size_t n);

  // Constructs and takes the derivative of a vector of the form:
  //   [ (1/0! dt^0), (1/1! dt^1), (1/2! dt^2) ... ]'
  // The derivative can be efficiently easily calculated by prepadding zeros
  // to the front of the vector and shifting to the right. This follows from
  // from the structure of the time vector.
  //
  // See the theory documentation for further details.
  Eigen::MatrixXd TimeVector(
      const size_t polynomial_order, 
      const size_t derivative_order, 
      const double time);
}
