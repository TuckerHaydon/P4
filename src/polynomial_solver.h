// Author: Tucker Haydon

#pragma once

#include <vector>
#include <memory>

#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "polynomial_bounds.h"

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
    void SetQuadraticCost(Eigen::SparseMatrix<T>& sparse_quadratic_mat);

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
        const T dt);

    // Sets the upper and lower bound vectors for the equality and continuity
    // constraints.
    template <class T>
    void SetConstraints(
        const std::vector<T>& times,
        Eigen::Matrix<T, Eigen::Dynamic, 1>& lower_bound_vec, 
        Eigen::Matrix<T, Eigen::Dynamic, 1>& upper_bound_vec,
        Eigen::SparseMatrix<T>& sparse_constraint_mat);
  
    private:
      Options options_;
      Workspace workspace_;
  }; 
}
