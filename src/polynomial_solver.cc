// Author: Tucker Haydon

#include <iostream>
#include <cstdlib>
#include <vector>

#include <Eigen/Dense>
#include <osqp.h>

#include "polynomial_solver.h"
#include "common.h"

namespace p4 {
  template <class T>
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> PolynomialSolver::QuadraticMatrix(
      const size_t polynomial_order,
      const size_t derivative_order,
      const T dt) {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> base_integrated_quadratic_matrix;
    base_integrated_quadratic_matrix.resize(polynomial_order + 1, polynomial_order + 1);
    base_integrated_quadratic_matrix.fill(T(0));
    // The number of coefficients is equal to the polynomial_order + 1
    for(size_t row = 0; row < polynomial_order + 1; ++row) {
      for(size_t col = 0; col < polynomial_order + 1; ++col) {
        base_integrated_quadratic_matrix(row, col) = 
          std::pow(dt, row + col + 1) 
          / T(Factorial(row))
          / T(Factorial(col))
          / T((row + col + 1));
      }
    }
  
    // Vector of ones
    Eigen::Matrix<T, Eigen::Dynamic, 1> ones_vec;
    ones_vec.resize(polynomial_order + 1 - derivative_order);
    ones_vec.fill(T(1));
  
    // Shift the matrix down rows
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> row_shift_mat;
    row_shift_mat.resize(polynomial_order + 1, polynomial_order + 1);
    row_shift_mat.fill(T(0));
    row_shift_mat.diagonal(-1*derivative_order) = ones_vec;
  
    // Shift the matrix right cols
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> col_shift_mat;
    col_shift_mat.resize(polynomial_order + 1, polynomial_order + 1);
    col_shift_mat.fill(T(0));
    col_shift_mat.diagonal(+1*derivative_order) = ones_vec;
  
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> integrated_quadratic_matrix;
    integrated_quadratic_matrix.resize(polynomial_order + 1, polynomial_order + 1);
    integrated_quadratic_matrix = row_shift_mat * base_integrated_quadratic_matrix * col_shift_mat;
  
    return integrated_quadratic_matrix;
  }

  template <class T>
  void PolynomialSolver::SetConstraints(
      Eigen::Matrix<T, Eigen::Dynamic, 1>& lower_bound_vec, 
      Eigen::Matrix<T, Eigen::Dynamic, 1>& upper_bound_vec,
      std::vector<Eigen::Triplet<T>>& constraint_triplets) {
    size_t constraint_idx = 0;
    for(size_t dimension_idx = 0; dimension_idx < this->workspace_.constants.num_dimensions; ++dimension_idx) { 
      for(size_t node_idx = 0; node_idx < this->workspace_.constants.num_nodes; ++node_idx) {
        for(size_t derivative_idx = 0; derivative_idx < this->workspace_.constants.num_params_per_segment_per_dim; ++derivative_idx) {

          // Equality Constraints
          for(const NodeEqualityBound& bound: this->workspace_.explicit_node_equality_bounds) {
            if(
                false == (bound.node_idx == node_idx) ||
                false == (bound.dimension_idx == dimension_idx) || 
                false == (bound.derivative_idx == derivative_idx)) {
              continue;
            }
            else {
              const T alpha = node_idx+1 < this->workspace_.constants.num_nodes 
                ? T(this->workspace_.times[node_idx + 1] - this->workspace_.times[node_idx]) : T(1);

              // Bounds. Scaled by alpha. See documentation.
              lower_bound_vec(constraint_idx) = T(bound.value) * std::pow(alpha, T(derivative_idx));
              upper_bound_vec(constraint_idx) = T(bound.value) * std::pow(alpha, T(derivative_idx));

              // Constraints
              size_t parameter_idx = 0 
                + derivative_idx 
                + this->workspace_.constants.num_params_per_node_per_dim * node_idx
                + this->workspace_.constants.num_params_per_node_per_dim * this->workspace_.constants.num_nodes * dimension_idx;
              constraint_triplets.emplace_back(constraint_idx, parameter_idx, T(1));

              constraint_idx++;
            }
          }

          // Node inequality bound constraints
          for(const NodeInequalityBound& bound: this->workspace_.explicit_node_inequality_bounds) {
            if(
                false == (bound.node_idx == node_idx) ||
                false == (bound.dimension_idx == dimension_idx) || 
                false == (bound.derivative_idx == derivative_idx)) {
              continue;
            }
            else {
              const T alpha = node_idx+1 < this->workspace_.constants.num_nodes 
                ? T(this->workspace_.times[node_idx + 1] - this->workspace_.times[node_idx]) : T(1);

              // Bounds. Scaled by alpha. See documentation.
              lower_bound_vec(constraint_idx) = T(bound.lower) * std::pow(alpha, T(derivative_idx));
              upper_bound_vec(constraint_idx) = T(bound.upper) * std::pow(alpha, T(derivative_idx));

              // Constraints
              size_t parameter_idx = 0 
                + derivative_idx 
                + this->workspace_.constants.num_params_per_node_per_dim * node_idx
                + this->workspace_.constants.num_params_per_node_per_dim * this->workspace_.constants.num_nodes * dimension_idx;
              constraint_triplets.emplace_back(constraint_idx, parameter_idx, T(1));

              constraint_idx++;
            }
          }
        }

        // Continuity constraints
        if(node_idx < this->workspace_.constants.num_segments) {
          const size_t num_continuity_constraints = this->workspace_.constants.continuity_order + 1;
          const double delta_t = 1.0;
          const T alpha_k = T(this->workspace_.times[node_idx + 1] - this->workspace_.times[node_idx]);
          const T alpha_kp1 = node_idx + 2 < this->workspace_.constants.num_nodes 
            ? T(this->workspace_.times[node_idx + 2] - this->workspace_.times[node_idx + 1]) : T(1.0);


          for(size_t continuity_idx = 0; continuity_idx < num_continuity_constraints; ++continuity_idx) {
            // Bounds
            lower_bound_vec(constraint_idx) = T(0);
            upper_bound_vec(constraint_idx) = T(0);

            // Constraints. Scaled by alpha. See documentation.
            // Propagate the current node
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> segment_propagation_coefficients;
            segment_propagation_coefficients.resize(1, this->workspace_.constants.num_params_per_segment_per_dim);
            segment_propagation_coefficients.fill(T(0));
            segment_propagation_coefficients 
              = TimeVector(this->workspace_.constants.polynomial_order, continuity_idx, delta_t).transpose().cast<T>()
              / std::pow(alpha_k, T(continuity_idx));

            // Minus the next node
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> segment_terminal_coefficients;
            segment_terminal_coefficients.resize(1, this->workspace_.constants.num_params_per_segment_per_dim);
            segment_terminal_coefficients.fill(T(0));
            segment_terminal_coefficients(0,continuity_idx) = T(-1) / std::pow(alpha_kp1, T(continuity_idx));

            size_t current_segment_idx = 0 
              // Get to the right dimension
              + this->workspace_.constants.num_params_per_node_per_dim 
              * this->workspace_.constants.num_nodes 
              * dimension_idx
              // Get to the right node
              + this->workspace_.constants.num_params_per_node_per_dim 
              * node_idx;
            size_t next_segment_idx = 0
              // Get to the right dimension
              + this->workspace_.constants.num_params_per_node_per_dim 
              * this->workspace_.constants.num_nodes 
              * dimension_idx
              // Get to the right node
              + this->workspace_.constants.num_params_per_node_per_dim 
              * (node_idx + 1);

            for(size_t param_idx = 0; param_idx < this->workspace_.constants.num_params_per_segment_per_dim; ++param_idx) {
              constraint_triplets.emplace_back(
                  constraint_idx, 
                  current_segment_idx + param_idx, 
                  segment_propagation_coefficients(0, param_idx));
              // TODO: Just insert one terminal constraint
              constraint_triplets.emplace_back(
                  constraint_idx, 
                  next_segment_idx + param_idx, 
                  segment_terminal_coefficients(0, param_idx));
            }

            constraint_idx++;
          }
        }
      }
    }

    // Include start- and end-points in segment constraints. When constraining
    // a segment, also constrain the endpoints of the segment to the same
    // value. If this were not the case, the following situation could occur:
    // the start endpoint is constrained to -2, but the following segment is
    // constrained above zero. Clearly, there is no smooth solution that
    // permits this. 
    // 
    // Add two to account for the endpoints and then remove one to convert
    // from the number of points the the number of segments. Divide the
    // segment length (1) by the number of segments to get the length of each
    // intermediate segment.
    const double dt = 1.0 / (this->workspace_.constants.num_intermediate_points + 2 - 1);

    // Segment lower bound constraints
    for(const SegmentInequalityBound& bound: this->workspace_.explicit_segment_inequality_bounds) {
      const T alpha = T(this->workspace_.times[bound.segment_idx+1] - this->workspace_.times[bound.segment_idx]);

      // point_idx == intermediate_point_idx
      // Add 2 for start and end points
      for(size_t point_idx = 0; point_idx < this->workspace_.constants.num_intermediate_points+2; ++point_idx)  {
        // Bounds
        lower_bound_vec(constraint_idx) = T(-SegmentInequalityBound::INFTY);
        upper_bound_vec(constraint_idx) = T(bound.value) * std::pow(alpha, T(bound.derivative_idx));

        // Time at a specific point
        double time = point_idx * dt;

        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> segment_propagation_coefficients;
        segment_propagation_coefficients.resize(1, this->workspace_.constants.num_params_per_segment_per_dim);
        segment_propagation_coefficients.fill(T(0));
        segment_propagation_coefficients 
          = TimeVector(this->workspace_.constants.polynomial_order, bound.derivative_idx, time).transpose().cast<T>();

        for(size_t dimension_idx = 0; dimension_idx < this->workspace_.constants.num_dimensions; ++dimension_idx) {
          Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> transform_coefficients;
          transform_coefficients.resize(1, this->workspace_.constants.num_params_per_segment_per_dim);
          transform_coefficients.fill(T(0));
          transform_coefficients = T(bound.mapping(dimension_idx, 0))
            * segment_propagation_coefficients;

          size_t current_segment_idx = 0 
            // Get to the right dimension
            + this->workspace_.constants.num_params_per_node_per_dim 
            * this->workspace_.constants.num_nodes 
            * dimension_idx
            // Get to the right node
            + this->workspace_.constants.num_params_per_segment_per_dim 
            * bound.segment_idx;

          for(size_t param_idx = 0; param_idx < this->workspace_.constants.num_params_per_segment_per_dim; ++param_idx) {
            constraint_triplets.emplace_back(
                constraint_idx, 
                current_segment_idx + param_idx, 
                transform_coefficients(0, param_idx));
          }
        }

        constraint_idx++;
      }
    }
  }

  template <class T>
  void PolynomialSolver::SetQuadraticCost(
      std::vector<Eigen::Triplet<T>>& quadratic_triplets) {
    const T delta_t = T(1.0);
    const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> quadratic_matrix = 
      this->QuadraticMatrix<T>(
          T(this->workspace_.constants.polynomial_order), 
          T(this->workspace_.constants.derivative_order),
          delta_t);

    for(size_t dimension_idx = 0; dimension_idx < this->workspace_.constants.num_dimensions; ++dimension_idx) {
      // No cost for final node
      for(size_t node_idx = 0; node_idx < this->workspace_.constants.num_nodes - 1; ++node_idx) {
        const size_t parameter_idx = 0
          // Get to the right dimension
          + this->workspace_.constants.num_params_per_node_per_dim 
          * this->workspace_.constants.num_nodes 
          * dimension_idx
          // Get to the right node
          + this->workspace_.constants.num_params_per_node_per_dim 
          * node_idx;
        for(size_t row = 0; row < this->workspace_.constants.num_params_per_node_per_dim; ++row) {
          for(size_t col = 0; col < this->workspace_.constants.num_params_per_node_per_dim; ++col) { 
            quadratic_triplets.emplace_back(
                T(row + parameter_idx), 
                T(col + parameter_idx), 
                quadratic_matrix(row,col)
                );
          }
        }
      }
    }
  }

  bool PolynomialSolver::Setup(
      const std::vector<double>& times,
      const std::vector<NodeEqualityBound>& explicit_node_equality_bounds,
      const std::vector<NodeInequalityBound>& explicit_node_inequality_bounds,
      const std::vector<SegmentInequalityBound>& explicit_segment_inequality_bounds) {

    // Checks
    if(times.size() < 2) {
      std::cerr << "PolynomialSolver::Setup -- Time vector must have a size greater than one." << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // Cache user input in the workspace
    this->workspace_.times = times;
    this->workspace_.explicit_node_equality_bounds = explicit_node_equality_bounds;
    this->workspace_.explicit_node_inequality_bounds = explicit_node_inequality_bounds;
    this->workspace_.explicit_segment_inequality_bounds = explicit_segment_inequality_bounds;

    // Cache constants in the workspace
    this->workspace_.constants.num_dimensions = 
      this->options_.num_dimensions;
    this->workspace_.constants.polynomial_order = 
      this->options_.polynomial_order;
    this->workspace_.constants.derivative_order = 
      this->options_.derivative_order;
    this->workspace_.constants.continuity_order = 
      this->options_.continuity_order;
    this->workspace_.constants.num_intermediate_points = 
      this->options_.num_intermediate_points;
    this->workspace_.constants.num_nodes = 
      times.size();
    this->workspace_.constants.num_segments = 
      this->workspace_.constants.num_nodes - 1;
    this->workspace_.constants.num_params_per_node_per_dim = 
      this->workspace_.constants.polynomial_order + 1;
    this->workspace_.constants.num_params_per_segment_per_dim = 
      this->workspace_.constants.polynomial_order + 1;
    this->workspace_.constants.num_params_per_node = 
      this->workspace_.constants.num_dimensions * 
      this->workspace_.constants.num_params_per_node_per_dim;
    this->workspace_.constants.num_params_per_segment = 
      this->workspace_.constants.num_dimensions * 
      this->workspace_.constants.num_params_per_segment_per_dim;
    this->workspace_.constants.total_num_params = 
      this->workspace_.constants.num_params_per_node * 
      this->workspace_.constants.num_nodes;

    // Explicit constraints are provided
    // Add two to the number of intermediate points due to the implicit endpoint
    // constraints
    const size_t num_explicit_constraints = 0
      + this->workspace_.explicit_node_equality_bounds.size() 
      + this->workspace_.explicit_node_inequality_bounds.size() 
      + this->workspace_.explicit_segment_inequality_bounds.size() 
      * (this->workspace_.constants.num_intermediate_points+2);

    // Implicit constraints are continuity constraints
    const size_t num_implicit_constraints = 
      this->workspace_.constants.num_segments
      *(this->workspace_.constants.continuity_order+1)
      *this->workspace_.constants.num_dimensions;

    this->workspace_.constants.num_constraints = num_explicit_constraints + num_implicit_constraints;

    // Allocate constraints
    this->workspace_.lower_bound_vec.resize(this->workspace_.constants.num_constraints);
    this->workspace_.upper_bound_vec.resize(this->workspace_.constants.num_constraints);
    this->workspace_.sparse_constraint_mat = Eigen::SparseMatrix<double>(
        this->workspace_.constants.num_constraints, 
        this->workspace_.constants.total_num_params);

    // Fill constraints
    std::vector<Eigen::Triplet<double>> constraint_triplets;
    this->SetConstraints<double>(
        this->workspace_.lower_bound_vec, 
        this->workspace_.upper_bound_vec, 
        constraint_triplets);

    this->workspace_.sparse_constraint_mat.setFromTriplets(
        constraint_triplets.begin(), 
        constraint_triplets.end());

    // Allocate quadratic matrix
    this->workspace_.sparse_quadratic_mat = Eigen::SparseMatrix<double>(
        this->workspace_.constants.total_num_params, 
        this->workspace_.constants.total_num_params);

    // Fill quadratic matrix
    std::vector<Eigen::Triplet<double>> quadratic_triplets;
    this->SetQuadraticCost<double>(quadratic_triplets);
    this->workspace_.sparse_quadratic_mat.setFromTriplets(
        quadratic_triplets.begin(), 
        quadratic_triplets.end());

    // Workspace is now set up
    this->workspace_.setup = true;
    return true;
  }


  PolynomialSolver::Solution PolynomialSolver::Run() {
    // Convert to solver data types
    csc* P = nullptr;
    csc* A = nullptr;

    Eigen2OSQP(this->workspace_.sparse_quadratic_mat, P);
    Eigen2OSQP(this->workspace_.sparse_constraint_mat, A);

    c_float q[this->workspace_.constants.total_num_params];
    for(size_t param_idx = 0; param_idx < this->workspace_.constants.total_num_params; ++param_idx) {
      q[param_idx] = 0;
    }

    c_float l[this->workspace_.constants.num_constraints], u[this->workspace_.constants.num_constraints];
    for(size_t row_idx = 0; row_idx < this->workspace_.constants.num_constraints; ++row_idx) {
      l[row_idx] = this->workspace_.lower_bound_vec(row_idx);
      u[row_idx] = this->workspace_.upper_bound_vec(row_idx);
    }

    // Run the solver
    // Allocate and populate data
    std::shared_ptr<OSQPData> data = std::shared_ptr<OSQPData>(
        (OSQPData *)c_malloc(sizeof(OSQPData)),
        [](OSQPData* data) {
          c_free(data->A);
          c_free(data->P);
        });
    data->n = this->workspace_.constants.total_num_params;
    data->m = this->workspace_.constants.num_constraints;
    data->P = P;
    data->q = q;
    data->A = A;
    data->l = l;
    data->u = u;

    // Allocate and prepare workspace
    // Workspace shared pointer requires custom destructor
    PolynomialSolver::Solution solution;
    solution.constants = this->workspace_.constants;
    solution.data      = data;
    solution.workspace = std::shared_ptr<OSQPWorkspace>(
        osqp_setup(data.get(), &this->options_.osqp_settings),
        [](OSQPWorkspace* workspace) { 
          osqp_cleanup(workspace);
        });

    // Solve
    osqp_solve(solution.workspace.get());

    // Return the solution
    return solution;
  }

  void PolynomialSolver::Options::Check() {
    if(this->num_dimensions < 1) {
      std::cerr << "PolynomialSolver::Options::Check -- Number of dimensions must be greater than zero." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  std::vector<std::vector<Eigen::VectorXd>> PolynomialSolver::Solution::Coefficients() const {
    std::vector<std::vector<Eigen::VectorXd>> coefficients;

    coefficients.resize(this->constants.num_dimensions);
    for(size_t dimension_idx = 0; dimension_idx < this->constants.num_dimensions; ++dimension_idx) {
      coefficients[dimension_idx].resize(this->constants.num_nodes);
      for(size_t node_idx = 0; node_idx < this->constants.num_nodes; ++node_idx) {
        coefficients[dimension_idx][node_idx].resize(this->constants.num_params_per_node_per_dim);
        for(size_t coefficient_idx = 0; coefficient_idx < this->constants.num_params_per_node_per_dim; ++coefficient_idx) {
          const size_t parameter_idx = 0
            // Get to the right dimension
            + this->constants.num_params_per_node_per_dim 
            * this->constants.num_nodes 
            * dimension_idx
            // Get to the right node
            + this->constants.num_params_per_node_per_dim 
            * node_idx
            // Get to the right parameter idx
            + coefficient_idx;

          coefficients[dimension_idx][node_idx](coefficient_idx)
            = this->workspace->solution->x[parameter_idx];
        }
      }
    }
    return coefficients;
  }

  Eigen::VectorXd PolynomialSolver::Solution::Coefficients(
      const size_t dimension_idx, 
      const size_t node_idx) const {

    Eigen::VectorXd coefficients;
    coefficients.resize(this->constants.num_params_per_node_per_dim);
    for(size_t coefficient_idx = 0; coefficient_idx < this->constants.num_params_per_node_per_dim; ++coefficient_idx) {
      const size_t parameter_idx = 0
        // Get to the right dimension
        + this->constants.num_params_per_node_per_dim 
        * this->constants.num_nodes 
        * dimension_idx
        // Get to the right node
        + this->constants.num_params_per_node_per_dim 
        * node_idx
        // Get to the right parameter idx
        + coefficient_idx;

      coefficients(coefficient_idx)
        = this->workspace->solution->x[parameter_idx];
    }

    return coefficients;
  }
}

