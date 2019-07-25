// Author: Tucker Haydon

#include <iostream>
#include <cstdlib>
#include <vector>

#include <Eigen/Dense>
#include <osqp.h>

#include "polynomial_solver.h"
#include "common.h"

namespace p4 {
  namespace {
    // Helper structure that contains pre-computed constants
    struct Constants {   
      size_t num_dimensions;
      size_t polynomial_order;
      size_t derivative_order;
      size_t continuity_order;
      size_t num_intermediate_points;
      size_t num_nodes;
      size_t num_segments;
      size_t num_params_per_node_per_dim;
      size_t num_params_per_segment_per_dim;
      size_t num_params_per_node;
      size_t num_params_per_segment;
      size_t total_num_params;
      size_t num_constraints;
    };

    // Generates a square matrix that is the integrated form of d^n/dt^n [p(x)'p(x)].
    // The derivative of this matrix can be easily calculated by computing the
    // zeroth derivative of the matrix, padding the first n rows and columns
    // with zeros, and shifting the matrix down and to the right by n
    // rows/columns.
    //
    // See the theory documentation for further details.
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
            / Factorial(row) 
            / Factorial(col) 
            / (row + col + 1);
        }
      }
    
      // Vector of ones
      Eigen::MatrixXd ones_vec;
      ones_vec.resize(polynomial_order + 1 - derivative_order, 1);
      ones_vec.fill(1);
    
      // Shift the matrix down rows
      Eigen::MatrixXd row_shift_mat;
      row_shift_mat.resize(polynomial_order + 1, polynomial_order + 1);
      row_shift_mat.fill(0);
      row_shift_mat.diagonal(-1*derivative_order) = ones_vec;
    
      // Shift the matrix right cols
      Eigen::MatrixXd col_shift_mat;
      col_shift_mat.resize(polynomial_order + 1, polynomial_order + 1);
      col_shift_mat.fill(0);
      col_shift_mat.diagonal(+1*derivative_order) = ones_vec;
    
      Eigen::MatrixXd integrated_quadratic_matrix;
      integrated_quadratic_matrix.resize(polynomial_order + 1, polynomial_order + 1);
      integrated_quadratic_matrix = row_shift_mat * base_integrated_quadratic_matrix * col_shift_mat;
    
      return integrated_quadratic_matrix;
    }

    // Sets the upper and lower bound vectors for the equality and continuity
    // constraints.
    void SetConstraints(
        const Constants& constants,
        const std::vector<double>& times,
        const std::vector<NodeEqualityBound>& explicit_node_equality_bounds, 
        const std::vector<NodeInequalityBound>& explicit_node_inequality_bounds,
        const std::vector<SegmentInequalityBound>& explicit_segment_inequality_bounds,
        Eigen::MatrixXd& lower_bound_vec, 
        Eigen::MatrixXd& upper_bound_vec,
        std::vector<Eigen::Triplet<double>>& constraint_triplets
        ) {
      size_t constraint_idx = 0;
      for(size_t dimension_idx = 0; dimension_idx < constants.num_dimensions; ++dimension_idx) { 
        for(size_t node_idx = 0; node_idx < constants.num_nodes; ++node_idx) {
          for(size_t derivative_idx = 0; derivative_idx < constants.num_params_per_segment_per_dim; ++derivative_idx) {

            // Equality Constraints
            for(const NodeEqualityBound& bound: explicit_node_equality_bounds) {
              if(
                  false == (bound.node_idx == node_idx) ||
                  false == (bound.dimension_idx == dimension_idx) || 
                  false == (bound.derivative_idx == derivative_idx)) {
                continue;
              }
              else {
                const double alpha = node_idx+1 < constants.num_nodes ? (times[node_idx + 1] - times[node_idx]) : 1;

                // Bounds. Scaled by alpha. See documentation.
                lower_bound_vec(constraint_idx,0) = bound.value * std::pow(alpha, derivative_idx);
                upper_bound_vec(constraint_idx,0) = bound.value * std::pow(alpha, derivative_idx);

                // Constraints
                size_t parameter_idx = 0 
                  + derivative_idx 
                  + constants.num_params_per_node_per_dim * node_idx
                  + constants.num_params_per_node_per_dim * constants.num_nodes * dimension_idx;
                constraint_triplets.emplace_back(constraint_idx, parameter_idx, 1);

                constraint_idx++;
              }
            }

            // Node inequality bound constraints
            for(const NodeInequalityBound& bound: explicit_node_inequality_bounds) {
              if(
                  false == (bound.node_idx == node_idx) ||
                  false == (bound.dimension_idx == dimension_idx) || 
                  false == (bound.derivative_idx == derivative_idx)) {
                continue;
              }
              else {
                const double alpha = node_idx+1 < constants.num_nodes ? (times[node_idx + 1] - times[node_idx]) : 1;

                // Bounds. Scaled by alpha. See documentation.
                lower_bound_vec(constraint_idx,0) = bound.lower * std::pow(alpha, derivative_idx);
                upper_bound_vec(constraint_idx,0) = bound.upper * std::pow(alpha, derivative_idx);

                // Constraints
                size_t parameter_idx = 0 
                  + derivative_idx 
                  + constants.num_params_per_node_per_dim * node_idx
                  + constants.num_params_per_node_per_dim * constants.num_nodes * dimension_idx;
                constraint_triplets.emplace_back(constraint_idx, parameter_idx, 1);

                constraint_idx++;
              }
            }
          }

          // Continuity constraints
          if(node_idx < constants.num_segments) {
            const size_t num_continuity_constraints = constants.continuity_order + 1;
            constexpr double delta_t = 1.0;
            const double alpha_k = times[node_idx + 1] - times[node_idx];
            const double alpha_kp1 = node_idx + 2 < constants.num_nodes ? times[node_idx + 2] - times[node_idx + 1] : 1.0;


            for(size_t continuity_idx = 0; continuity_idx < num_continuity_constraints; ++continuity_idx) {
              // Bounds
              lower_bound_vec(constraint_idx,0) = 0;
              upper_bound_vec(constraint_idx,0) = 0;

              // Constraints. Scaled by alpha. See documentation.
              // Propagate the current node
              Eigen::MatrixXd segment_propagation_coefficients;
              segment_propagation_coefficients.resize(1, constants.num_params_per_segment_per_dim);
              segment_propagation_coefficients.fill(0);
              segment_propagation_coefficients 
                = TimeVector(constants.polynomial_order, continuity_idx, delta_t).transpose()
                / std::pow(alpha_k, continuity_idx);

              // Minus the next node
              Eigen::MatrixXd segment_terminal_coefficients;
              segment_terminal_coefficients.resize(1, constants.num_params_per_segment_per_dim);
              segment_terminal_coefficients.fill(0);
              segment_terminal_coefficients(0,continuity_idx) = -1 / std::pow(alpha_kp1, continuity_idx);

              size_t current_segment_idx = 0 
                // Get to the right dimension
                + constants.num_params_per_node_per_dim * constants.num_nodes * dimension_idx
                // Get to the right node
                + constants.num_params_per_node_per_dim * node_idx;
              size_t next_segment_idx = 0
                // Get to the right dimension
                + constants.num_params_per_node_per_dim * constants.num_nodes * dimension_idx
                // Get to the right node
                + constants.num_params_per_node_per_dim * (node_idx + 1);

              for(size_t param_idx = 0; param_idx < constants.num_params_per_segment_per_dim; ++param_idx) {
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
      const double dt = 1.0 / (constants.num_intermediate_points + 2 - 1);

      // Segment lower bound constraints
      for(const SegmentInequalityBound& bound: explicit_segment_inequality_bounds) {
        const double alpha = times[bound.segment_idx+1] - times[bound.segment_idx];

        // point_idx == intermediate_point_idx
        // Add 2 for start and end points
        for(size_t point_idx = 0; point_idx < constants.num_intermediate_points+2; ++point_idx)  {
          // Bounds
          lower_bound_vec(constraint_idx,0) = -SegmentInequalityBound::INFTY;
          upper_bound_vec(constraint_idx,0) = bound.value * std::pow(alpha, bound.derivative_idx);

          // Time at a specific point
          double time = point_idx * dt;

          Eigen::MatrixXd segment_propagation_coefficients;
          segment_propagation_coefficients.resize(1, constants.num_params_per_segment_per_dim);
          segment_propagation_coefficients.fill(0);
          segment_propagation_coefficients 
            = TimeVector(constants.polynomial_order, bound.derivative_idx, time).transpose();

          for(size_t dimension_idx = 0; dimension_idx < constants.num_dimensions; ++dimension_idx) {
            Eigen::MatrixXd transform_coefficients;
            transform_coefficients.resize(1, constants.num_params_per_segment_per_dim);
            transform_coefficients.fill(0);
            transform_coefficients = bound.mapping(dimension_idx, 0) 
              * segment_propagation_coefficients;

            size_t current_segment_idx = 0 
              // Get to the right dimension
              + constants.num_params_per_node_per_dim * constants.num_nodes * dimension_idx
              // Get to the right node
              + constants.num_params_per_segment_per_dim * bound.segment_idx;

            for(size_t param_idx = 0; param_idx < constants.num_params_per_segment_per_dim; ++param_idx) {
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

    void SetQuadraticCost(
        const Constants& constants,
        std::vector<Eigen::Triplet<double>>& quadratic_triplets) {
      const double delta_t = 1.0;
      const Eigen::MatrixXd quadratic_matrix = QuadraticMatrix(constants.polynomial_order, constants.derivative_order, delta_t);

      for(size_t dimension_idx = 0; dimension_idx < constants.num_dimensions; ++dimension_idx) {
        // No cost for final node
        for(size_t node_idx = 0; node_idx < constants.num_nodes - 1; ++node_idx) {
          const size_t parameter_idx = 0
            // Get to the right dimension
            + constants.num_params_per_node_per_dim * constants.num_nodes * dimension_idx
            // Get to the right node
            + constants.num_params_per_node_per_dim * node_idx;
          for(size_t row = 0; row < constants.num_params_per_node_per_dim; ++row) {
            for(size_t col = 0; col < constants.num_params_per_node_per_dim; ++col) { 
              quadratic_triplets.emplace_back(
                  row + parameter_idx, 
                  col + parameter_idx, 
                  quadratic_matrix(row,col)
                  );
            }
          }
        }
      }
    }

  }


  PolynomialSolver::Solution PolynomialSolver::Run(
      const std::vector<double>& times,
      const std::vector<NodeEqualityBound>& explicit_node_equality_bounds,
      const std::vector<NodeInequalityBound>& explicit_node_inequality_bounds,
      const std::vector<SegmentInequalityBound>& explicit_segment_inequality_bounds) {

    this->options_.Check();

    if(times.size() < 2) {
      std::cerr << "PolynomialSolver::Run -- Time vector must have a size greater than one." << std::endl;
      std::exit(EXIT_FAILURE);
    }

    Constants constants;
    constants.num_dimensions = this->options_.num_dimensions;
    constants.polynomial_order = this->options_.polynomial_order;
    constants.derivative_order = this->options_.derivative_order;
    constants.continuity_order = this->options_.continuity_order;
    constants.num_intermediate_points = this->options_.num_intermediate_points;
    constants.num_nodes = times.size();
    constants.num_segments = constants.num_nodes - 1;
    constants.num_params_per_node_per_dim = constants.polynomial_order + 1;
    constants.num_params_per_segment_per_dim = constants.polynomial_order + 1;
    constants.num_params_per_node = constants.num_dimensions * constants.num_params_per_node_per_dim;
    constants.num_params_per_segment = constants.num_dimensions * constants.num_params_per_segment_per_dim;
    constants.total_num_params = constants.num_params_per_node * constants.num_nodes;

    // Explicit constraints are provided
    const size_t num_explicit_constraints = 0
      + explicit_node_equality_bounds.size() 
      + explicit_node_inequality_bounds.size() 
      + explicit_segment_inequality_bounds.size() * (constants.num_intermediate_points+2);

    // Implicit constraints are continuity constraints
    const size_t num_implicit_constraints = constants.num_segments*(constants.continuity_order+1)*constants.num_dimensions;

    constants.num_constraints = num_explicit_constraints + num_implicit_constraints;

    /*
     * CONSTRAINTS
     */
    Eigen::MatrixXd lower_bound_vec, upper_bound_vec;
    lower_bound_vec.resize(constants.num_constraints, 1);
    upper_bound_vec.resize(constants.num_constraints, 1);

    std::vector<Eigen::Triplet<double>> constraint_triplets;
    SetConstraints(
        constants, 
        times,
        explicit_node_equality_bounds,
        explicit_node_inequality_bounds,
        explicit_segment_inequality_bounds,
        lower_bound_vec, 
        upper_bound_vec, 
        constraint_triplets);

    // Triplets to sparse mat
    Eigen::SparseMatrix<double> sparse_constraint_mat(
        constants.num_constraints, 
        constants.total_num_params);
    sparse_constraint_mat.setFromTriplets(
        constraint_triplets.begin(), 
        constraint_triplets.end());

    /*
     * QUADRATIC MATRIX
     */
    std::vector<Eigen::Triplet<double>> quadratic_triplets;
    SetQuadraticCost(constants, quadratic_triplets);

    // Triplets to sparse mat
    Eigen::SparseMatrix<double> sparse_quadratic_mat(
        constants.total_num_params, 
        constants.total_num_params);
    sparse_quadratic_mat.setFromTriplets(
        quadratic_triplets.begin(), 
        quadratic_triplets.end());

    /*
     * CONVERT EIGEN TO OSQP
     */
    csc* P = nullptr;
    csc* A = nullptr;

    Eigen2OSQP(sparse_quadratic_mat, P);
    Eigen2OSQP(sparse_constraint_mat, A);

    c_float q[constants.total_num_params];
    for(size_t param_idx = 0; param_idx < constants.total_num_params; ++param_idx) {
      q[param_idx] = 0;
    }

    c_float l[constants.num_constraints], u[constants.num_constraints];
    for(size_t row_idx = 0; row_idx < constants.num_constraints; ++row_idx) {
      l[row_idx] = lower_bound_vec(row_idx, 0);
      u[row_idx] = upper_bound_vec(row_idx, 0);
    }

    /*
     * RUN THE SOLVER
     */
    // Allocate and populate data
    std::shared_ptr<OSQPData> data = std::shared_ptr<OSQPData>(
        (OSQPData *)c_malloc(sizeof(OSQPData)),
        [](OSQPData* data) {
          c_free(data->A);
          c_free(data->P);
        });
    data->n = constants.total_num_params;
    data->m = constants.num_constraints;
    data->P = P;
    data->q = q;
    data->A = A;
    data->l = l;
    data->u = u;

    // Allocate and prepare workspace
    // Workspace shared pointer requires custom destructor
    PolynomialSolver::Solution solution;
    solution.num_dimensions   = constants.num_dimensions;
    solution.polynomial_order = constants.polynomial_order;
    solution.num_nodes        = constants.num_nodes;
    solution.data = data;
    solution.workspace =  std::shared_ptr<OSQPWorkspace>(
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
    const size_t num_segments                   = this->num_nodes - 1;
    const size_t num_params_per_node_per_dim    = this->polynomial_order + 1;
    const size_t num_params_per_segment_per_dim = this->polynomial_order + 1;
    const size_t num_params_per_node            = this->num_dimensions * num_params_per_node_per_dim;
    const size_t num_params_per_segment         = this->num_dimensions * num_params_per_segment_per_dim;
    const size_t total_num_params               = this->num_nodes * num_params_per_node;

    std::vector<std::vector<Eigen::VectorXd>> coefficients;

    coefficients.resize(this->num_dimensions);
    for(size_t dimension_idx = 0; dimension_idx < this->num_dimensions; ++dimension_idx) {
      coefficients[dimension_idx].resize(this->num_nodes);
      for(size_t node_idx = 0; node_idx < this->num_nodes; ++node_idx) {
        coefficients[dimension_idx][node_idx].resize(num_params_per_node_per_dim);
        for(size_t coefficient_idx = 0; coefficient_idx < num_params_per_node_per_dim; ++coefficient_idx) {
          const size_t parameter_idx = 0
            // Get to the right dimension
            + num_params_per_node_per_dim * this->num_nodes * dimension_idx
            // Get to the right node
            + num_params_per_node_per_dim * node_idx
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
    const size_t num_segments                   = this->num_nodes - 1;
    const size_t num_params_per_node_per_dim    = this->polynomial_order + 1;
    const size_t num_params_per_segment_per_dim = this->polynomial_order + 1;
    const size_t num_params_per_node            = this->num_dimensions * num_params_per_node_per_dim;
    const size_t num_params_per_segment         = this->num_dimensions * num_params_per_segment_per_dim;
    const size_t total_num_params               = this->num_nodes * num_params_per_node;

    Eigen::VectorXd coefficients;
    coefficients.resize(num_params_per_node_per_dim);
    for(size_t coefficient_idx = 0; coefficient_idx < num_params_per_node_per_dim; ++coefficient_idx) {
      const size_t parameter_idx = 0
        // Get to the right dimension
        + num_params_per_node_per_dim * this->num_nodes * dimension_idx
        // Get to the right node
        + num_params_per_node_per_dim * node_idx
        // Get to the right parameter idx
        + coefficient_idx;

      coefficients(coefficient_idx)
        = this->workspace->solution->x[parameter_idx];
    }

    return coefficients;
  }
}

