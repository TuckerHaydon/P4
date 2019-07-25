// Author: Tucker Haydon

#pragma once

#include <vector>
#include <memory>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "polynomial_bounds.h"
#include "common.h"

namespace p4 {
  /* Class for solving piecewise polynomial fitting & minimization problems.
   *
   * Given a polynomial of the following form:
   *   p_1(t) = c_10 (1/0! t^0) + c_11 (1/1! t^1) + c_12 (1/2! t^2) + c_13 (1/3! t^3) + ...
   *   p_2(t) = c_20 (1/0! t^0) + c_21 (1/1! t^1) + c_22 (1/2! t^2) + c_23 (1/3! t^3) + ...
   *   p_3(t) = c_30 (1/0! t^0) + c_31 (1/1! t^1) + c_32 (1/2! t^2) + c_33 (1/3! t^3) + ...
   *   ...
   *
   * finds the minimum of the following cost function:
   *   x^T P x
   *
   * subject to continuity and path constraints.
   *
   * The polynomials are found with a quadratic programming solver. OSQP was
   * chosen to solve this problem. OSQP can efficiently solve sparse QP
   * problems. OSQP requires the problem to be formulated as:
   *   argmin
   *     x^T P x
   *   subject to
   *     l <= Ax <= u
   *
   * Notes: 
   * 1) polynomial_order must be 3 or more orders greater than derivative_order
   * 2) continuity_order must be less than polynomial_order.
   * 3) state vector is ordered first by polynomial index, then by segment index,
   *    and finally by dimension index.
   * 4) see theory documention for further information
   */
  class PolynomialSolver {
    public:
      // Options to configure the solver with
      struct Options {
        // Required. Standard options
        size_t num_dimensions   = 0;
        size_t polynomial_order = 0;
        size_t derivative_order = 0;
        size_t continuity_order = 0;

        // Optional. Number of intermediate points for segment inequality constraints
        size_t num_intermediate_points = 20;

        // Optional. Solver settings.
        OSQPSettings osqp_settings;
  
        // Constructor
        Options() {
          osqp_set_default_settings(&(this->osqp_settings));
        }

        // Evaluate whether the options are valid
        void Check();
      };

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

        Constants() {}
      };

      // Structure wrapping important information about the OSQP solution
      struct Solution {
        // Heap-allocated shared pointer to an OSQPWorkspace instance. Must
        // define a custom destructor for cleanup.
        // Important notes:
        //   a) Do not use workspace->data. It returns bad data. I don't know
        //   why. Instead, use an OSQPData instance included below.
        // Important includes:
        //   a) workspace.info.obj_val: the optimal cost of the optimization
        //   problem: J = 0.5 * x' * P * x
        //   b) workspace.solution: solution and lagrange multipliers
        // Resources: 
        //   a) https://osqp.org/docs/interfaces/cc++#workspace
        std::shared_ptr<OSQPWorkspace> workspace = nullptr;

        // Heap-allocated shared pointer to an OSQPData instance. Must define a
        // custom destructor for cleanup.
        // Resources:
        //   a) https://osqp.org/docs/interfaces/cc++#data
        std::shared_ptr<OSQPData> data = nullptr;

        // Constants structure helpful for decoding the OSQP data later.
        Constants constants;

        // Constructor
        Solution() {};

        // Reshapes the coefficients of the OSQP solution into a more usable
        // format. Returns a 3D data structure with the following format:
        // [dimension_idx][segment_idx][coefficient_idx]
        std::vector<std::vector<Eigen::VectorXd>> Coefficients() const;

        // Returns an Eigen vector containing the coefficients for a specified
        // dimension and segment index.
        Eigen::VectorXd Coefficients(
            const size_t dimension_idx, 
            const size_t node_idx) const;
      };

      // Structure to cache data in for the Setup() function.
      struct Workspace {
        // Supplied by user
        std::vector<double> times;
        std::vector<NodeEqualityBound> explicit_node_equality_bounds;
        std::vector<NodeInequalityBound> explicit_node_inequality_bounds;
        std::vector<SegmentInequalityBound> explicit_segment_inequality_bounds;

        // Filled in
        Constants constants;

        // Constraints
        Eigen::Matrix<double, Eigen::Dynamic, 1> lower_bound_vec;
        Eigen::Matrix<double, Eigen::Dynamic, 1> upper_bound_vec;
        Eigen::SparseMatrix<double> sparse_constraint_mat;

        // Quadratic matrix
        Eigen::SparseMatrix<double> sparse_quadratic_mat;

        // Run() should only be called if this is true
        bool setup = false;

        Workspace() {}
      };

      // Constructor
      PolynomialSolver(const Options& options = Options())
        : options_(options) {}

      // Setup translates input data into structures to be used by the QP
      // solver. Depending on the size of the problem, this function may be
      // expensive as it allocates space for large data types.
      //
      // Setup must be called before Run().
      //
      // Returns true if solver is set up. Returns false if an error occurred.
      bool Setup(
          const std::vector<double>& times,
          const std::vector<NodeEqualityBound>& node_equality_bounds,
          const std::vector<NodeInequalityBound>& node_inequality_bounds,
          const std::vector<SegmentInequalityBound>& segment_inequality_bounds);
  
      // Run the QP solver
      Solution Run();

    template <class T>
    void SetQuadraticCost(Eigen::SparseMatrix<T>& sparse_quadratic_mat) const;

    // Generates a square matrix that is the integrated form of d^n/dt^n [p(x)'p(x)].
    // The derivative of this matrix can be easily calculated by computing the
    // zeroth derivative of the matrix, padding the first n rows and columns
    // with zeros, and shifting the matrix down and to the right by n
    // rows/columns.
    //
    // See the theory documentation for further details.
    template <class T>
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> QuadraticMatrix(
        const size_t polynomial_order,
        const size_t derivative_order,
        const T dt) const;

    // Sets the upper and lower bound vectors for the equality and continuity
    // constraints.
    template <class T>
    void SetConstraints(
        const std::vector<T>& times,
        Eigen::Matrix<T, Eigen::Dynamic, 1>& lower_bound_vec, 
        Eigen::Matrix<T, Eigen::Dynamic, 1>& upper_bound_vec,
        Eigen::SparseMatrix<T>& sparse_constraint_mat) const;
  
    private:
      Options options_;
      Workspace workspace_;
  }; 

  /*******************************
   * Inline Template Definitions *
   *******************************/
  template <class T>
  Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> PolynomialSolver::QuadraticMatrix(
      const size_t polynomial_order,
      const size_t derivative_order,
      const T dt) const {
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> base_integrated_quadratic_matrix;
    base_integrated_quadratic_matrix.resize(polynomial_order + 1, polynomial_order + 1);
    base_integrated_quadratic_matrix.fill(T(0));
    // The number of coefficients is equal to the polynomial_order + 1
    for(size_t row = 0; row < polynomial_order + 1; ++row) {
      for(size_t col = 0; col < polynomial_order + 1; ++col) {
        base_integrated_quadratic_matrix(row, col) = 
          pow(dt, row + col + 1) 
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
      const std::vector<T>& times,
      Eigen::Matrix<T, Eigen::Dynamic, 1>& lower_bound_vec, 
      Eigen::Matrix<T, Eigen::Dynamic, 1>& upper_bound_vec,
      Eigen::SparseMatrix<T>& sparse_constraint_mat) const {

    // Resize and allocate space for output
    std::vector<Eigen::Triplet<T>> constraint_triplets;
    lower_bound_vec.resize(this->workspace_.constants.num_constraints);
    upper_bound_vec.resize(this->workspace_.constants.num_constraints);
    sparse_constraint_mat = Eigen::SparseMatrix<T>(
        this->workspace_.constants.num_constraints,
        this->workspace_.constants.total_num_params);

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
                ? times[node_idx + 1] - times[node_idx] : T(1);

              // Bounds. Scaled by alpha. See documentation.
              lower_bound_vec(constraint_idx) = T(bound.value) * pow(alpha, T(derivative_idx));
              upper_bound_vec(constraint_idx) = T(bound.value) * pow(alpha, T(derivative_idx));

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
                ? times[node_idx + 1] - times[node_idx] : T(1);

              // Bounds. Scaled by alpha. See documentation.
              lower_bound_vec(constraint_idx) = T(bound.lower) * pow(alpha, T(derivative_idx));
              upper_bound_vec(constraint_idx) = T(bound.upper) * pow(alpha, T(derivative_idx));

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
          const T alpha_k = times[node_idx + 1] - times[node_idx];
          const T alpha_kp1 = node_idx + 2 < this->workspace_.constants.num_nodes 
            ? times[node_idx + 2] - times[node_idx + 1] : T(1.0);


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
              / pow(alpha_k, T(continuity_idx));

            // Minus the next node
            Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> segment_terminal_coefficients;
            segment_terminal_coefficients.resize(1, this->workspace_.constants.num_params_per_segment_per_dim);
            segment_terminal_coefficients.fill(T(0));
            segment_terminal_coefficients(0,continuity_idx) = T(-1) / pow(alpha_kp1, T(continuity_idx));

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
      const T alpha = times[bound.segment_idx+1] - times[bound.segment_idx];

      // point_idx == intermediate_point_idx
      // Add 2 for start and end points
      for(size_t point_idx = 0; point_idx < this->workspace_.constants.num_intermediate_points+2; ++point_idx)  {
        // Bounds
        lower_bound_vec(constraint_idx) = T(-SegmentInequalityBound::INFTY);
        upper_bound_vec(constraint_idx) = T(bound.value) * pow(alpha, T(bound.derivative_idx));

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

    sparse_constraint_mat.setFromTriplets(
        constraint_triplets.begin(), 
        constraint_triplets.end());
  }

  template <class T>
  void PolynomialSolver::SetQuadraticCost(Eigen::SparseMatrix<T>& sparse_mat) const {
    // Allocate space for sparse matrix
    sparse_mat = Eigen::SparseMatrix<T>(
        this->workspace_.constants.total_num_params,
        this->workspace_.constants.total_num_params);

    std::vector<Eigen::Triplet<T>> quadratic_triplets;
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

    sparse_mat.setFromTriplets(
        quadratic_triplets.begin(), 
        quadratic_triplets.end());
  }
}
