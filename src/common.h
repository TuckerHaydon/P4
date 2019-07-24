// Author: Tucker Haydon

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <osqp.h>
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

  // Converts an en eigen sparse matrix into an OSQP sparse matrix
  // Reference: https://github.com/robotology/osqp-eigen
  void Eigen2OSQP(
      const Eigen::SparseMatrix<double>& eigen_sparse_mat,
      csc*& osqp_mat);

  // Converts an en osqp sparse matrix to an eigen sparse matrix
  // Reference: https://github.com/robotology/osqp-eigen
  void OSQP2Eigen(
      const csc* const & osqp_mat,
      Eigen::SparseMatrix<double>& eigen_sparse_mat);
}
