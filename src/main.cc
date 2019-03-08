// Author: Tucker Haydon

// #include "OsqpEigen/OsqpEigen.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <osqp.h>

#include <iostream>
#include <cstdlib>
#include <vector>
#include <queue>

template <size_t T>
struct PathConstraint {
  size_t index;
  Eigen::Matrix<double, T, 1> constraint;
  
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  PathConstraint(
      const size_t index_,
      const Eigen::Matrix<double, T, 1>& constraint_)
  : index(index_),
    constraint(constraint_) {}
};

// Reference: https://github.com/robotology/osqp-eigen
template<typename T>
bool Eigen2OSQP(const Eigen::SparseMatrix<T>& eigen_mat,
                csc*& osqp_mat) {
  // get number of row, columns and nonZeros from Eigen SparseMatrix
  c_int rows   = eigen_mat.rows();
  c_int cols   = eigen_mat.cols();
  c_int num_nz = eigen_mat.nonZeros();

  // get inner and outer index
  const int* innerIndexPtr    = eigen_mat.innerIndexPtr();
  const int* outerIndexPtr    = eigen_mat.outerIndexPtr();
  const int* innerNonZerosPtr = eigen_mat.innerNonZeroPtr();

  // get nonzero values
  const T* valuePtr = eigen_mat.valuePtr();

  // Allocate memory for csc matrix
  if(osqp_mat != nullptr){
    std::cerr << "osqp_mat pointer is not a null pointer! " << std::endl;
    return false;
  }

  osqp_mat = csc_spalloc(rows, cols, num_nz, 1, 0);

  int innerOsqpPosition = 0;
  for(int k = 0; k < cols; ++k) {
      if (eigen_mat.isCompressed()) {
          osqp_mat->p[k] = static_cast<c_int>(outerIndexPtr[k]);
      } else {
          if (k == 0) {
              osqp_mat->p[k] = 0;
          } else {
              osqp_mat->p[k] = osqp_mat->p[k-1] + innerNonZerosPtr[k-1];
          }
      }
      for (typename Eigen::SparseMatrix<T>::InnerIterator it(eigen_mat,k); it; ++it) {
          osqp_mat->i[innerOsqpPosition] = static_cast<c_int>(it.row());
          osqp_mat->x[innerOsqpPosition] = static_cast<c_float>(it.value());
          innerOsqpPosition++;
      }
  }
  osqp_mat->p[static_cast<int>(cols)] = static_cast<c_int>(innerOsqpPosition);

  assert(innerOsqpPosition == num_nz);

  return true;
}

size_t factorial(size_t n) {
  return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
}

template <size_t DIMENSION>
void Path2PVA(
    const std::vector<PathConstraint<DIMENSION>>& pos_constraints,
    const std::vector<PathConstraint<DIMENSION>>& vel_constraints,
    const std::vector<PathConstraint<DIMENSION>>& acc_constraints,
    const std::vector<double> times
    ) {

  // Input requirements
  if( 2 > pos_constraints.size() || 
      2 > vel_constraints.size() ||
      2 > acc_constraints.size() || 
      2 > times.size()) {
    std::cerr << "::Path2PVA -- Inputs malformed." << std::endl;
    std::exit(EXIT_FAILURE);
  }

  // TODO: Check sorted by increasing index

  constexpr size_t MIN_DERIVATIVE_IDX        = 2; // 5 is snap
  constexpr size_t POLYNOMIAL_SIZE           = 2; // Order of polynomial. Usually 7
  constexpr size_t NUM_CONTINUOUS_PARAMETERS = (POLYNOMIAL_SIZE + 1); // Usually 5
  constexpr double WIGGLE                    = 1e-4;
  const size_t num_nodes                     = times.size();
  const size_t num_intermediate_nodes        = times.size() - 2;
  const size_t num_segments                  = times.size() - 1;
  const size_t num_parameters_per_segment    = DIMENSION * (POLYNOMIAL_SIZE + 1);
  const size_t num_parameters                = DIMENSION * (POLYNOMIAL_SIZE + 1) * num_nodes;

  // The start and end are constrained by three derivatives
  // Each segment is constrained by NUM_CONTINUOUS_DERIVATIVES
  // The last segment is constrained by only three derivatives
  const size_t num_constraints = 2*DIMENSION*3 + DIMENSION*NUM_CONTINUOUS_PARAMETERS*(num_segments - 1);

  /* NOTES
   * 1) Constraints are ordered in the following way:
   *  a) Node
   *  b) Continuity
   *  c) SFC
   */

  /* 
   * SET THE UPPER AND LOWER BOUNDS
   */
  // Start, End, Continuity
  Eigen::MatrixXd lower_bound, upper_bound;
  lower_bound.resize(num_constraints, 1);
  upper_bound.resize(num_constraints, 1);

  size_t bound_idx = 0;

  { // Node bounds
    std::queue<PathConstraint<DIMENSION>> pos_queue, vel_queue, acc_queue;
    for(const auto& el: pos_constraints) { pos_queue.push(el); };
    for(const auto& el: vel_constraints) { vel_queue.push(el); };
    for(const auto& el: acc_constraints) { acc_queue.push(el); };

    for(size_t node_idx = 0; node_idx < num_nodes-1; ++node_idx) {
      if(node_idx == pos_queue.front().index) {
        lower_bound.block<DIMENSION,1>(bound_idx, 0) = pos_queue.front().constraint - Eigen::Matrix<double, DIMENSION, 1>::Ones() * WIGGLE;
        upper_bound.block<DIMENSION,1>(bound_idx, 0) = pos_queue.front().constraint + Eigen::Matrix<double, DIMENSION, 1>::Ones() * WIGGLE;
        bound_idx += DIMENSION;
        pos_queue.pop();
      }
      if(node_idx == vel_queue.front().index) {
        lower_bound.block<DIMENSION,1>(bound_idx, 0) = vel_queue.front().constraint - Eigen::Matrix<double, DIMENSION, 1>::Ones() * WIGGLE;
        upper_bound.block<DIMENSION,1>(bound_idx, 0) = vel_queue.front().constraint + Eigen::Matrix<double, DIMENSION, 1>::Ones() * WIGGLE;
        bound_idx += DIMENSION;
        vel_queue.pop();
      }
      if(node_idx == acc_queue.front().index) {
        lower_bound.block<DIMENSION,1>(bound_idx, 0) = acc_queue.front().constraint - Eigen::Matrix<double, DIMENSION, 1>::Ones() * WIGGLE;
        upper_bound.block<DIMENSION,1>(bound_idx, 0) = acc_queue.front().constraint + Eigen::Matrix<double, DIMENSION, 1>::Ones() * WIGGLE;
        bound_idx += DIMENSION;
        vel_queue.pop();
      } 
    }
  }

  { // Continuity bounds
    // Note: The number of continuity bounds is unique for the final segment
    for(size_t segment_idx = 0; segment_idx < num_segments; ++segment_idx) {
      const size_t num_continuity_constraints = 
        (segment_idx == num_segments - 1) ? 3 : NUM_CONTINUOUS_PARAMETERS;

      lower_bound.block(bound_idx, 0, num_continuity_constraints, 1) 
        = Eigen::MatrixXd::Ones(num_continuity_constraints,1) * WIGGLE * -1;

      upper_bound.block(bound_idx, 0, num_continuity_constraints, 1) 
        = Eigen::MatrixXd::Ones(num_continuity_constraints,1) * WIGGLE * +1;
      bound_idx += num_continuity_constraints;
    }
  }

  // SFC bounds
  // TODO

  /*
   * SET THE CONSTRAINT MATRIX
   */
  Eigen::MatrixXd dense_constraint_matrix;
  dense_constraint_matrix.resize(num_constraints, num_parameters);
  dense_constraint_matrix.fill(0);
  size_t constraint_idx = 0;

  { // Node Constraints
    // Helper constants
    const size_t pos_idx = 0;
    const size_t vel_idx = DIMENSION;
    const size_t acc_idx = 2*DIMENSION;

    std::queue<PathConstraint<DIMENSION>> pos_queue, vel_queue, acc_queue;
    for(const auto& el: pos_constraints) { pos_queue.push(el); };
    for(const auto& el: vel_constraints) { vel_queue.push(el); };
    for(const auto& el: acc_constraints) { acc_queue.push(el); };

    for(size_t node_idx = 0; node_idx < num_nodes-1; ++node_idx) {
      if(node_idx == pos_queue.front().index) {
        dense_constraint_matrix.block<DIMENSION,DIMENSION>(constraint_idx, pos_idx) = Eigen::Matrix<double, DIMENSION, DIMENSION>::Identity();
        pos_queue.pop();
        constraint_idx += DIMENSION;
      }
      if(node_idx == vel_queue.front().index) {
        dense_constraint_matrix.block<DIMENSION,DIMENSION>(constraint_idx, vel_idx) = Eigen::Matrix<double, DIMENSION, DIMENSION>::Identity();
        vel_queue.pop();
        constraint_idx += DIMENSION;
      }
      if(node_idx == acc_queue.front().index) {
        dense_constraint_matrix.block<DIMENSION,DIMENSION>(constraint_idx, acc_idx) = Eigen::Matrix<double, DIMENSION, DIMENSION>::Identity();
        acc_queue.pop();
        constraint_idx += DIMENSION;
      }
    }
  }

  { // Continuity Constraints
    // Note: The number of continuity bounds is unique for the final segment
    for(size_t segment_idx = 0; segment_idx < num_segments; ++segment_idx) {
      const size_t segment_start_idx = num_parameters_per_segment * segment_idx;
      const size_t segment_end_idx = segment_start_idx + (num_parameters_per_segment * 1);
      // Time per segment is scaled to 1 for numerical stability. Must un-scale at end.
      const double delta_t = 1.0;
      const size_t num_continuity_constraints = 
        (segment_idx == num_segments - 1) ? 3 : NUM_CONTINUOUS_PARAMETERS;
      for(size_t continuity_idx = 0; continuity_idx < num_continuity_constraints; ++continuity_idx) {
        const size_t segment_parameter_start_idx = segment_start_idx + continuity_idx * DIMENSION;
        const size_t segment_parameter_end_idx = segment_end_idx + continuity_idx * DIMENSION;

        // Coefficients for propagating the starting node to the end of the
        // segment using a zero-order-hold assumption
        Eigen::MatrixXd start_segment_propagation_coefficients;
        start_segment_propagation_coefficients.resize(DIMENSION, num_parameters_per_segment);
        start_segment_propagation_coefficients.fill(0);
        for(size_t derivative_idx = continuity_idx; derivative_idx <= POLYNOMIAL_SIZE; ++derivative_idx) {
            start_segment_propagation_coefficients.block<DIMENSION,1>(0,derivative_idx) = 
              Eigen::Matrix<double, DIMENSION, 1>::Ones() 
              * std::pow(delta_t, derivative_idx - continuity_idx) 
              / factorial(derivative_idx - continuity_idx);
        }

        // Negative of the end node, requiring the propagation and the end to
        // be the same
        Eigen::MatrixXd end_segment_propagation_coefficients;
        end_segment_propagation_coefficients.resize(DIMENSION, num_parameters_per_segment);
        end_segment_propagation_coefficients.fill(0);
        end_segment_propagation_coefficients.block<DIMENSION,1>(0,continuity_idx) = 
          Eigen::Matrix<double, DIMENSION, 1>::Ones() * -1;

        dense_constraint_matrix.block<DIMENSION,num_parameters_per_segment>
          (constraint_idx, segment_start_idx) = start_segment_propagation_coefficients;
        dense_constraint_matrix.block<DIMENSION,num_parameters_per_segment>
          (constraint_idx, segment_end_idx) = end_segment_propagation_coefficients;
        constraint_idx += DIMENSION;
      }
    }
  }

  { // SFC Constraints
    // TODO
  }

  /*
   * SET THE QUADRATIC COST MATRIX
   */
  Eigen::MatrixXd dense_quadratic_cost_matrix;
  dense_quadratic_cost_matrix.resize(num_parameters, num_parameters);
  dense_quadratic_cost_matrix.fill(0);
  for(size_t segment_idx = 0; segment_idx < num_segments; ++segment_idx) {
    const size_t min_idx = num_parameters_per_segment * segment_idx + DIMENSION*MIN_DERIVATIVE_IDX;
    dense_quadratic_cost_matrix.block<DIMENSION,DIMENSION>(min_idx, min_idx) = Eigen::Matrix<double, DIMENSION, DIMENSION>::Identity();
  }

  /*
   * CONVERT EIGEN MATRICES TO OSQP MATRICES
   */
  const Eigen::SparseMatrix<double> sparse_constraint_matrix 
    = dense_constraint_matrix.sparseView(1, 1e-20);
  csc* constraint_mat = nullptr;
  Eigen2OSQP(sparse_constraint_matrix, constraint_mat);

  const Eigen::SparseMatrix<double> sparse_quadratic_cost_matrix 
    = dense_quadratic_cost_matrix.sparseView(1, 1e-20);
  csc* quadratic_mat = nullptr;
  Eigen2OSQP(sparse_quadratic_cost_matrix, quadratic_mat);

  std::vector<c_float> linear_mat(num_parameters, 0.0);

  std::vector<c_float> lower_mat(num_constraints), upper_mat(num_constraints);
  for(size_t row_idx = 0; row_idx < num_constraints; ++row_idx) {
    lower_mat[row_idx] = lower_bound(row_idx, 0);
    upper_mat[row_idx] = upper_bound(row_idx, 0);
  }


  std::cout << dense_quadratic_cost_matrix << std::endl;
  std::cout << "" << std::endl;
  std::cout << dense_constraint_matrix << std::endl;
  std::cout << "" << std::endl;
  std::cout << lower_bound.transpose() << std::endl;
  std::cout << upper_bound.transpose() << std::endl;

  /*
   * SOLVER
   */
  // Reference: https://osqp.org/docs/examples/demo.html
  // Problem settings
  OSQPSettings* settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

  // Structures
  OSQPWorkspace* work;
  OSQPData* data;  

  // Populate data
  data = (OSQPData*)c_malloc(sizeof(OSQPData));
  data->n = num_parameters;
  data->m = num_constraints;
  data->P = quadratic_mat;
  data->q = linear_mat.data();
  data->A = constraint_mat;
  data->l = lower_mat.data();
  data->u = upper_mat.data();

  // Define Solver settings as default
  osqp_set_default_settings(settings);
  settings->alpha = 1.0; 

  // Setup workspace
  work = osqp_setup(data, settings);

  // Solve Problem
  osqp_solve(work);

  std::cout << work->info->status << std::endl;

  // Cleanup
  osqp_cleanup(work);
  c_free(data->A);
  c_free(data->P);
  c_free(data);
  c_free(settings);

  return;
}


int main() { 
  // std::vector<PathConstraint> pos_constraint;
  // pos_constraint.emplace_back(0, Eigen::Matrix<double, 4, 1>(0,0,0,0));
  // pos_constraint.emplace_back(1, Eigen::Matrix<double, 4, 1>(1,0,0,0));

  // std::vector<PathConstraint> vel_constraint;
  // vel_constraint.emplace_back(0, Eigen::Matrix<double, 4, 1>(1,0,0,0));
  // vel_constraint.emplace_back(1, Eigen::Matrix<double, 4, 1>(0,0,0,0));

  // std::vector<PathConstraint> acc_constraint;
  // acc_constraint.emplace_back(0, Eigen::Matrix<double, 4, 1>(0,0,0,0));
  // acc_constraint.emplace_back(1, Eigen::Matrix<double, 4, 1>(0,0,0,0));

  // std::vector<double> times = {0, 0.5};
  // Path2PVA(pos_constraint, vel_constraint, acc_constraint, times);

  std::vector<PathConstraint<1>> pos_constraint;
  pos_constraint.emplace_back(0, Eigen::Matrix<double, 1, 1>(0));
  pos_constraint.emplace_back(1, Eigen::Matrix<double, 1, 1>(1));
  pos_constraint.emplace_back(2, Eigen::Matrix<double, 1, 1>(2));

  std::vector<PathConstraint<1>> vel_constraint;
  vel_constraint.emplace_back(0, Eigen::Matrix<double, 1, 1>(1));
  vel_constraint.emplace_back(1, Eigen::Matrix<double, 1, 1>(0));
  vel_constraint.emplace_back(2, Eigen::Matrix<double, 1, 1>(1));

  std::vector<PathConstraint<1>> acc_constraint;
  acc_constraint.emplace_back(0, Eigen::Matrix<double, 1, 1>(0));
  acc_constraint.emplace_back(1, Eigen::Matrix<double, 1, 1>(0));
  acc_constraint.emplace_back(2, Eigen::Matrix<double, 1, 1>(0));

  std::vector<double> times = {0, 0.5, 1};
  Path2PVA<1>(pos_constraint, vel_constraint, acc_constraint, times);

  return EXIT_SUCCESS;
}
