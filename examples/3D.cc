// Author: Tucker Haydon

#include <cstdlib>
#include <Eigen/Dense>

#include "polynomial_solver.h"
#include "polynomial_sampler.h"

using namespace mediation_layer;

int main(int argc, char** argv) {

  const std::vector<double> times = {0, 1, 2};

  // Equality bounds paramater order is:
  // 1) Dimension index
  // 2) Node index
  // 3) Derivative index
  // 4) Value
  const std::vector<EqualityBound> equality_bounds = {
    // The first node must constrain position, velocity, and acceleration
    EqualityBound(0,0,0,0),
    EqualityBound(1,0,0,0),
    EqualityBound(2,0,0,0),
    EqualityBound(0,0,1,0),
    EqualityBound(1,0,1,0),
    EqualityBound(2,0,1,0),
    EqualityBound(0,0,2,0),
    EqualityBound(1,0,2,0),
    EqualityBound(2,0,2,0),

    // The second node constrains position
    EqualityBound(0,1,0,1),
    EqualityBound(1,1,0,0),
    EqualityBound(2,1,0,0),

    // The third node constrains position
    EqualityBound(0,2,0,1),
    EqualityBound(1,2,0,0),
    EqualityBound(2,2,0,1),

    // The fourth node constrains position
    EqualityBound(0,3,0,0),
    EqualityBound(1,3,0,0),
    EqualityBound(2,3,0,0),
  };

  const std::vector<LowerBound> lower_bounds = {};
  const std::vector<UpperBound> upper_bounds = {};

  PolynomialSolver::Options solver_options;
  solver_options.num_dimensions = 3;     // 3D
  solver_options.polynomial_order = 7;   // Fit an 7th-order polynomial
  solver_options.continuity_order = 4;   // Require continuity to the 4th order
  solver_options.derivative_order = 4;   // Minimize the 4th order (snap)
  solver_options.polish = true;          // Polish the solution

  PolynomialSolver solver(solver_options);
  const PolynomialPath path
    = solver.Run(times, equality_bounds, upper_bounds, lower_bounds);

  PolynomialSampler::Options sampler_options;
  sampler_options.frequency = 20;

  PolynomialSampler sampler(sampler_options);
  sampler.Run(times, path);

  return EXIT_SUCCESS;
}
