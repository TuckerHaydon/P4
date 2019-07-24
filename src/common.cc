// Author: Tucker Haydon

#include "common.h"

#include <iostream>

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

  void Eigen2OSQP(
      const Eigen::SparseMatrix<double>& eigen_sparse_mat,
      csc*& osqp_mat) {

    // Get number of row, columns and nonZeros from Eigen SparseMatrix
    c_int rows   = eigen_sparse_mat.rows();
    c_int cols   = eigen_sparse_mat.cols();
    c_int num_nz = eigen_sparse_mat.nonZeros();
  
    // get inner and outer index
    const int* inner_index_ptr     = eigen_sparse_mat.innerIndexPtr();
    const int* outer_index_ptr     = eigen_sparse_mat.outerIndexPtr();
    const int* inner_non_zeros_ptr = eigen_sparse_mat.innerNonZeroPtr();
  
    // get nonzero values
    const double* value_ptr = eigen_sparse_mat.valuePtr();
  
    // Allocate memory for csc matrix
    if(osqp_mat != nullptr){
      std::cerr << "osqp_mat pointer is not a null pointer! " << std::endl;
      std::exit(EXIT_FAILURE);
    }
  
    osqp_mat = csc_spalloc(rows, cols, num_nz, 1, 0);
  
    int inner_osqp_position = 0;
    for(int k = 0; k < cols; ++k) {
        if (eigen_sparse_mat.isCompressed()) {
            osqp_mat->p[k] = static_cast<c_int>(outer_index_ptr[k]);
        } else {
            if (k == 0) {
                osqp_mat->p[k] = 0;
            } else {
                osqp_mat->p[k] = osqp_mat->p[k-1] + inner_non_zeros_ptr[k-1];
            }
        }
        for (typename Eigen::SparseMatrix<double>::InnerIterator it(eigen_sparse_mat,k); it; ++it) {
            osqp_mat->i[inner_osqp_position] = static_cast<c_int>(it.row());
            osqp_mat->x[inner_osqp_position] = static_cast<c_float>(it.value());
            inner_osqp_position++;
        }
    }
    osqp_mat->p[static_cast<int>(cols)] = static_cast<c_int>(inner_osqp_position);
  }

  void OSQP2Eigen(
      const csc* const & osqp_mat,
      Eigen::SparseMatrix<double>& eigen_sparse_mat) {

    // Get the number of rows and columns
    int rows = osqp_mat->m;
    int cols = osqp_mat->n;

    // Get the triplets from the csc matrix
    std::vector<Eigen::Triplet<c_float>> triplet_list;

    // Get row and column data
    c_int* inner_index_ptr = osqp_mat->i;
    c_int* outer_index_ptr = osqp_mat->p;

    // Get values data
    c_float* value_ptr = osqp_mat->x;
    c_int num_non_zero =  osqp_mat->p[osqp_mat->n];

    // Populate the tripletes vector
    int column=0;
    int row;
    c_float value;

    triplet_list.resize(num_non_zero);
    for(int i = 0; i < num_non_zero; ++i) {
        row = inner_index_ptr[i];
        value = value_ptr[i];

        while(i >= outer_index_ptr[column+1]) {
            column++;
        }

        triplet_list[i] = Eigen::Triplet<c_float>(row, column, value);
    }

    triplet_list.erase(triplet_list.begin() + num_non_zero, triplet_list.end());

    // resize the eigen matrix
    eigen_sparse_mat.resize(rows, cols);

    // set the eigen matrix from triplets
    eigen_sparse_mat.setFromTriplets(triplet_list.begin(), triplet_list.end());
  }
}
