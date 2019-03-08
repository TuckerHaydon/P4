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

Eigen::MatrixXd QuadraticMatrix(
    const size_t polynomial_order,
    const size_t derivative_order,
    const double dt) {
  Eigen::MatrixXd base_integrated_quadratic_matrix;
  base_integrated_quadratic_matrix.resize(polynomial_order + 1, polynomial_order + 1);
  base_integrated_quadratic_matrix.fill(0);
  for(size_t row = 0; row < polynomial_order + 1; ++row) {
    for(size_t col = 0; col < polynomial_order + 1; ++col) {
      base_integrated_quadratic_matrix(row, col) = 
        std::pow(dt, row + col + 1) 
        / factorial(row) 
        / factorial(col) 
        / (row + col + 1);
    }
  }

  Eigen::MatrixXd ones_vec;
  ones_vec.resize(polynomial_order + 1 - derivative_order, 1);
  ones_vec.fill(1);

  Eigen::MatrixXd row_shift_mat;
  row_shift_mat.resize(polynomial_order + 1, polynomial_order + 1);
  row_shift_mat.fill(0);
  row_shift_mat.diagonal(-1*derivative_order) = ones_vec;

  Eigen::MatrixXd col_shift_mat;
  col_shift_mat.resize(polynomial_order + 1, polynomial_order + 1);
  col_shift_mat.fill(0);
  col_shift_mat.diagonal(+1*derivative_order) = ones_vec;

  Eigen::MatrixXd integrated_quadratic_matrix;
  integrated_quadratic_matrix.resize(polynomial_order + 1, polynomial_order + 1);
  integrated_quadratic_matrix = row_shift_mat * base_integrated_quadratic_matrix * col_shift_mat;

  return integrated_quadratic_matrix;
}

Eigen::MatrixXd CoefficientVector(
    const size_t polynomial_order, 
    const size_t derivative_order, 
    const double dt) {
  Eigen::MatrixXd base_coefficient_vec;
  base_coefficient_vec.resize(polynomial_order + 1,1);
  for(size_t idx = 0; idx < polynomial_order + 1; ++idx) {
    base_coefficient_vec(idx, 0) = std::pow(dt, idx) / factorial(idx);
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

  constexpr size_t DERIVATIVE_ORDER          = 2;                      // 2 is acceleration, 4 is snap
  constexpr size_t POLYNOMIAL_ORDER          = DERIVATIVE_ORDER + 3; // Usually 7
  constexpr size_t CONTINUITY_ORDER          = 2;                      // Usually 5
  constexpr double WIGGLE                    = 0;
  const size_t num_nodes                     = times.size();
  const size_t num_intermediate_nodes        = times.size() - 2;
  const size_t num_segments                  = times.size() - 1;
  const size_t num_parameters_per_segment    = DIMENSION * (POLYNOMIAL_ORDER + 1);
  const size_t num_parameters_per_dimension  = (POLYNOMIAL_ORDER + 1);
  const size_t num_parameters                = DIMENSION * (POLYNOMIAL_ORDER + 1) * num_nodes;


  // The start and end are constrained by three derivatives
  // Each segment is constrained by NUM_CONTINUOUS_DERIVATIVES
  // The last segment is constrained by only three derivatives
  // Each intermediate node is constrained by something
  constexpr size_t NUM_NODAL_CONSTRAINTS_PER_DIMENSION  = 3;
  constexpr size_t NUM_NODAL_CONSTRAINTS                = NUM_NODAL_CONSTRAINTS_PER_DIMENSION * DIMENSION;

  const size_t num_constraints = 0
    + 2*NUM_NODAL_CONSTRAINTS                                       // Start,end
    + num_intermediate_nodes*NUM_NODAL_CONSTRAINTS                  // Intermediate, 3 is temporary
    + DIMENSION*CONTINUITY_ORDER*(num_segments - 1)                 // Continuity
    + NUM_NODAL_CONSTRAINTS;                                        // Final continuity segment

  /* NOTES
   * 1) Constraints are ordered in the following way:
   *  a) Node 
   *  b) Continuity
   *  c) SFC
   *
   * 2) Polynomial size must be greater than derivative idx by 3 or more
   * 3) Permutation of constraints is node, dimension, order. Example: All of the
   *    polynomial coefficients for x for the first node, then all the polynomial coefficients for
   *    y for the first node, the all the polynomial coefficients for x for the
   *    second node, etc
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
    // TODO: What if only intermediate position constraints?
    for(size_t node_idx = 0; node_idx < num_nodes; ++node_idx) {
      for(size_t dim = 0; dim < DIMENSION; ++dim) { 
        lower_bound(bound_idx,0) = pos_constraints[node_idx].constraint(dim);
        upper_bound(bound_idx,0) = pos_constraints[node_idx].constraint(dim);
        bound_idx++;

        lower_bound(bound_idx,0) = vel_constraints[node_idx].constraint(dim);
        upper_bound(bound_idx,0) = vel_constraints[node_idx].constraint(dim);
        bound_idx++;

        lower_bound(bound_idx,0) = acc_constraints[node_idx].constraint(dim);
        upper_bound(bound_idx,0) = acc_constraints[node_idx].constraint(dim);
        bound_idx++;
      }
    }
  }

  { // Continuity bounds
    // Note: The number of continuity bounds is unique for the final segment
    for(size_t segment_idx = 0; segment_idx < num_segments; ++segment_idx) {
      for(size_t dim = 0; dim < DIMENSION; ++dim) {
        const size_t num_continuity_constraints = 
          (segment_idx == num_segments - 1) ? 3 : CONTINUITY_ORDER;

        lower_bound.block(bound_idx, 0, num_continuity_constraints, 1) 
          = Eigen::MatrixXd::Ones(num_continuity_constraints,1) * WIGGLE * -1;

        upper_bound.block(bound_idx, 0, num_continuity_constraints, 1) 
          = Eigen::MatrixXd::Ones(num_continuity_constraints,1) * WIGGLE * +1;

        bound_idx += num_continuity_constraints;
      }
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
    // TODO: What if only position constraints?
    for(size_t node_idx = 0; node_idx < num_nodes; ++node_idx) {
      for(size_t dim = 0; dim < DIMENSION; ++dim) { 
        const size_t parameter_idx = node_idx * num_parameters_per_segment + dim * num_parameters_per_dimension;
        dense_constraint_matrix.block<3,3>(constraint_idx,parameter_idx) = Eigen::Matrix<double, 3, 3>::Identity();
        constraint_idx += 3;
      }
    }
  }

  { // Continuity Constraints
    for(size_t segment_idx = 0; segment_idx < num_segments; ++segment_idx) {
      const size_t this_segment_parameter_idx = num_parameters_per_segment * segment_idx;
      const size_t next_segment_parameter_idx = num_parameters_per_segment * (segment_idx + 1);

      // Time per segment is scaled to 1 for numerical stability. Must un-scale at end.
      const double delta_t = 1.0;

      for(size_t dim = 0; dim < DIMENSION; ++dim) {

        // Final segment is only constrained by PVA
        const size_t num_continuity_constraints = 
          (segment_idx == num_segments - 1) ? 3 : CONTINUITY_ORDER;

        // Continuity index is the index of the derivative whose continuity is
        // begin enforced
        for(size_t continuity_idx = 0; continuity_idx < num_continuity_constraints; ++continuity_idx) {
          // Propagate the current node
          Eigen::MatrixXd segment_propagation_coefficients;
          segment_propagation_coefficients.resize(1, num_parameters_per_dimension);
          segment_propagation_coefficients.fill(0);
          segment_propagation_coefficients 
            = CoefficientVector(POLYNOMIAL_ORDER, continuity_idx, delta_t).transpose();

          // Minus the next node
          Eigen::MatrixXd segment_terminal_coefficients;
          segment_terminal_coefficients.resize(1, num_parameters_per_dimension);
          segment_terminal_coefficients.fill(0);
          segment_terminal_coefficients(0,continuity_idx) = -1;

          size_t this_parameter_idx = this_segment_parameter_idx + dim * num_parameters_per_dimension;
          size_t next_parameter_idx = next_segment_parameter_idx + dim * num_parameters_per_dimension;

          dense_constraint_matrix.block<1,num_parameters_per_dimension>
            (constraint_idx, this_parameter_idx) = segment_propagation_coefficients;
          dense_constraint_matrix.block<1,num_parameters_per_dimension>
            (constraint_idx, next_parameter_idx) = segment_terminal_coefficients;

          constraint_idx += 1;
        }
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
  {
    const double delta_t = 1.0;
    const Eigen::MatrixXd quadratic_matrix = QuadraticMatrix(POLYNOMIAL_ORDER, DERIVATIVE_ORDER, delta_t);

    for(size_t segment_idx = 0; segment_idx < num_segments; ++segment_idx) {
      for(size_t dim = 0; dim < DIMENSION; ++dim) {
        dense_quadratic_cost_matrix.block(
            segment_idx*num_parameters_per_segment + dim*num_parameters_per_dimension,
            segment_idx*num_parameters_per_segment + dim*num_parameters_per_dimension,
            num_parameters_per_dimension, 
            num_parameters_per_dimension) = quadratic_matrix;
      }
    }
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
  settings->warm_start = false;
  settings->polish = true;

  // Setup workspace
  work = osqp_setup(data, settings);

  // Solve Problem
  osqp_solve(work);

  std::cout << work->info->status << std::endl;
  std::cout << "Run Time: " << work->info->run_time << " s" << std::endl;

  for(size_t solution_idx = 0; solution_idx < num_parameters; ++solution_idx) {
    std::cout << work->solution->x[solution_idx] << std::endl;
  }

  // Cleanup
  osqp_cleanup(work);
  c_free(data->A);
  c_free(data->P);
  c_free(data);
  c_free(settings);

  return;
}


int main() { 
  std::vector<PathConstraint<1>> pos_constraint;
  std::vector<PathConstraint<1>> vel_constraint;
  std::vector<PathConstraint<1>> acc_constraint;
  std::vector<double> times;

  pos_constraint.emplace_back(0, Eigen::Matrix<double, 1, 1>(0));
  vel_constraint.emplace_back(0, Eigen::Matrix<double, 1, 1>(0));
  acc_constraint.emplace_back(0, Eigen::Matrix<double, 1, 1>(0));

  pos_constraint.emplace_back(1, Eigen::Matrix<double, 1, 1>(1));
  vel_constraint.emplace_back(1, Eigen::Matrix<double, 1, 1>(0));
  acc_constraint.emplace_back(1, Eigen::Matrix<double, 1, 1>(0));

  pos_constraint.emplace_back(2, Eigen::Matrix<double, 1, 1>(2));
  vel_constraint.emplace_back(2, Eigen::Matrix<double, 1, 1>(0));
  acc_constraint.emplace_back(2, Eigen::Matrix<double, 1, 1>(0));

  times = {0,1,2};

  Path2PVA<1>(pos_constraint, vel_constraint, acc_constraint, times);

  // std::vector<PathConstraint<2>> pos_constraint;
  // std::vector<PathConstraint<2>> vel_constraint;
  // std::vector<PathConstraint<2>> acc_constraint;

  // pos_constraint.emplace_back(0, Eigen::Matrix<double, 2, 1>(0,0));
  // vel_constraint.emplace_back(0, Eigen::Matrix<double, 2, 1>(0,0));
  // acc_constraint.emplace_back(0, Eigen::Matrix<double, 2, 1>(0,0));

  // pos_constraint.emplace_back(1, Eigen::Matrix<double, 2, 1>(1,1));
  // vel_constraint.emplace_back(1, Eigen::Matrix<double, 2, 1>(0,0));
  // acc_constraint.emplace_back(1, Eigen::Matrix<double, 2, 1>(0,0));

  // std::vector<double> times = {0, 0.5};
  // Path2PVA<2>(pos_constraint, vel_constraint, acc_constraint, times);

  // std::vector<PathConstraint<3>> pos_constraint;
  // std::vector<PathConstraint<3>> vel_constraint;
  // std::vector<PathConstraint<3>> acc_constraint;
  // std::vector<double> times;

  // // pos_constraint.emplace_back(0, Eigen::Matrix<double, 3, 1>(0,0,0));
  // // vel_constraint.emplace_back(0, Eigen::Matrix<double, 3, 1>(0,0,0));
  // // acc_constraint.emplace_back(0, Eigen::Matrix<double, 3, 1>(0,0,0));

  // // pos_constraint.emplace_back(1, Eigen::Matrix<double, 3, 1>(1,1,1));
  // // vel_constraint.emplace_back(1, Eigen::Matrix<double, 3, 1>(0,0,0));
  // // acc_constraint.emplace_back(1, Eigen::Matrix<double, 3, 1>(0,0,0));

  // for(size_t idx = 0; idx < 100; ++idx) {
  //   pos_constraint.emplace_back(0, Eigen::Matrix<double, 3, 1>(idx, idx*idx, 3.0*idx));
  //   vel_constraint.emplace_back(1, Eigen::Matrix<double, 3, 1>(0,0,0));
  //   acc_constraint.emplace_back(2, Eigen::Matrix<double, 3, 1>(0,0,0));
  //   times.emplace_back(idx);
  // }
  // Path2PVA<3>(pos_constraint, vel_constraint, acc_constraint, times);

  return EXIT_SUCCESS;
}

// void test_order {
//   std::vector<PathConstraint<3>> pos_constraint;
//   std::vector<PathConstraint<3>> vel_constraint;
//   std::vector<PathConstraint<3>> acc_constraint;
// 
//   pos_constraint.emplace_back(0, Eigen::Matrix<double, 3, 1>(1,4,7));
//   vel_constraint.emplace_back(0, Eigen::Matrix<double, 3, 1>(2,5,8));
//   acc_constraint.emplace_back(0, Eigen::Matrix<double, 3, 1>(3,6,9));
// 
//   pos_constraint.emplace_back(1, Eigen::Matrix<double, 3, 1>(10,13,16));
//   vel_constraint.emplace_back(1, Eigen::Matrix<double, 3, 1>(11,14,17));
//   acc_constraint.emplace_back(1, Eigen::Matrix<double, 3, 1>(12,15,18));
// 
//   std::vector<double> times = {0, 0.5};
//   Path2PVA<3>(pos_constraint, vel_constraint, acc_constraint, times);
// }
