// Author: Tucker Haydon

#pragma once

#include <vector>
#include <memory>

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
        // Standard options
        size_t num_dimensions   = 0;
        size_t polynomial_order = 0;
        size_t derivative_order = 0;
        size_t continuity_order = 0;

        // Number of intermediate points for segment inequality constraints
        size_t num_intermediate_points = 20;

        // Solver settings. These are freed after
        OSQPSettings osqp_settings;
  
        Options() {
          osqp_set_default_settings(&(this->osqp_settings));
        }

        // Evaluate whether the options are valid
        void Check();
      };

      // Structure wrapping important information about the OSQP solution
      struct Solution {
        // Heap-allocated shared pointer to an OSQPWorkspace instance. Must
        // define a custom destructor for cleanup.
        // Important includes:
        //   a) workspace.info.obj_val: the optimal cost of the optimization
        //   problem: J = 0.5 * x' * P * x
        //   b) workspace.solution: solution and lagrange multipliers
        // Resources: 
        //   a) https://osqp.org/docs/interfaces/cc++#workspace
        std::shared_ptr<OSQPWorkspace> workspace = nullptr;

        // Number of dimensions
        size_t num_dimensions   = 0;
        // Order of piecewise polynomial
        size_t polynomial_order = 0;
        // Number of nodes (corresponds with the number of times)
        size_t num_nodes        = 0;

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

      PolynomialSolver(const Options& options = Options())
        : options_(options) {}
  
      Solution Run(
          const std::vector<double>& times,
          const std::vector<NodeEqualityBound>& node_equality_bounds,
          const std::vector<NodeInequalityBound>& node_inequality_bounds,
          const std::vector<SegmentInequalityBound>& segment_inequality_bounds);
  
    private:
      Options options_;
  }; 
}
