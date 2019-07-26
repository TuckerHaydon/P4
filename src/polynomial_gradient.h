// Author: Tucker Haydon

#pragma once

#include "polynomial_solver.h"

namespace p4 {
  // Class encapsulating the algorithm used to determine the gradient of the
  // following bi-level optimization problem:
  // argmin(x,y) 0.5 * x' P(y) x + q(y)' x + c(y)
  //   s.t. 
  //     Ay <= b
  //     Cy == d
  //     x = argmin(x) 0.5 * x' P x + q(y)' x + c(y)
  //       s.t.
  //         G(y) x <= h(y)
  //         L(y) x == m(y)
  //
  // The goal is to determine the partial derivative with respect to y of:
  //   f(x,y) = 0.5 * x' P(y) x + q(y)' x + c(y) 
  //
  // With y (waypoint arrival time) held fixed, x (polynomial coefficients) can
  // easily be found with a quadratic programming solution. Thus, the ultimate
  // goal is to perform a gradient descent over y, while solving for the
  // intermediate x, thus necessitating the partial derivative with respect to
  // y.
  //   f*(y) == f(x*,y)
  //
  // The gradient of f* is found with the following equation (see Sun 2018):
  //   grad(f*(y)) = lambda' partial_y(g(x*,y)) + mu' partial_y(h(x*,y)) +
  //                 partial_y(f(x*,y))
  // Where
  //   f*(y) == f(x*,y)
  //   g(x*,y) <= 0 : inequality contraints of sub-problem
  //   h(x*,y) == 0 : equality constraints of sub-problem
  //   lambda, mu   : lagrange multipliers of corresponding constraints
  //
  // The above section describes the general formulation. These are the
  // substitutions made for the specific problem at hand:
  //   P(y) = P : Constant due to the normalization of the time segments
  //   q(y) = 0 : No linear cost
  //   c(y) = ones() * y : penalize total time
  //   A = diag(-1,0) + diag(+1,1) : Difference adjacent times are require >= 0
  //   b = 0 : Used with A to ensure time is increasing
  //   C,y = 0: Enforce the initial time is zero
  //   g(x*,y) = G(y) x - h(y)
  //
  // Recall that the sub-level optimization problem required the following
  // constraint:
  //   l <= Nx <= u
  //
  // There are only inequality constraints. Equality constraints were enforced
  // by setting l == u. Thus, h(x*,y) = 0. Moreover, the normalization of the
  // waypoint arrival time to [0,1] removed the dependence of P(y) on y, but
  // introduced a time-dependence in the inequality constraints due to the alpha
  // term (see documentation in docs/):
  //   l(y) <= N(y)x <= u(y)
  //
  // Restructuring:
  //   -N(y)x + l(y) <= 0
  //    N(y)x - u(y) <= 0
  //
  // Thus:
  //   G(y) = [-N(y); N(y)]
  //   h(y) = [ l(y); u(y)]
  //
  // However, OSQP overloads the lagrange multipliers. There should be 2*m
  // lagrange multipliers, but OSQP returns only m. The kkt conditions require:
  //   lambda+ * ( N(y)x - u(y)) = 0
  //   lambda- * (-N(y)x + l(y)) = 0
  //
  // Where:
  //   lambda+ = max(lambda, 0)
  //   lambda- = min(y,0) * -1
  //
  // Resources:
  // 1) Fast UAV Trajectory Optimization using Bilevel Optimization with
  //    Analytical Gradients, Sun et. al., 2018
  class PolynomialGradient {
    public:
      struct Options {

        Options() {}
      };

      struct Solution {
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> gradient;

        Solution() {}
      };

      PolynomialGradient(const Options& options = Options())
        : options_(options) {}

      // Evaluate the time-gradient of the bi-level optimization problem.
      // Leverage google::ceres and its automatic differentiation engine to
      // determing the numeric derivatives.
      Solution Run(
          const std::vector<double>& initial_times,
          const std::shared_ptr<const PolynomialSolver>& solver,
          const PolynomialSolver::Solution& solver_solution);

    private:
      Options options_;
  };
}
