// Author: Tucker Haydon

#include "common.h"

namespace p4 {
  size_t Factorial(size_t n) {
    return (n == 1 || n == 0) ? 1 : Factorial(n - 1) * n;
  }

  Eigen::MatrixXd TimeVector(
      const size_t polynomial_order, 
      const size_t derivative_order, 
      const double time) {
    Eigen::MatrixXd base_coefficient_vec;
    base_coefficient_vec.resize(polynomial_order + 1,1);
    for(size_t idx = 0; idx < polynomial_order + 1; ++idx) {
      // std::pow(0,0) undefined. Define as 1.0.
      if(0 == idx && 0.0 == time) {
        base_coefficient_vec(idx, 0) = 1.0 / Factorial(idx);
      } else {
        base_coefficient_vec(idx, 0) = std::pow(time, idx) / Factorial(idx);
      }
    }
  
    Eigen::MatrixXd ones_vec;
    ones_vec.resize(polynomial_order + 1 - derivative_order, 1);
    ones_vec.fill(1);
  
    Eigen::MatrixXd shift_mat;
    shift_mat.resize(polynomial_order + 1, polynomial_order + 1);
    shift_mat.fill(0);
    shift_mat.diagonal(-1*derivative_order) = ones_vec;
  
    Eigen::MatrixXd coefficient_vec;
    coefficient_vec.resize(polynomial_order + 1, 1);
    coefficient_vec = shift_mat * base_coefficient_vec;
  
    return coefficient_vec;
  } 
}
