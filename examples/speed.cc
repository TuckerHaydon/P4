// Author: Tucker Haydon

#include <cstdlib>

#include <Eigen/Dense>

#include "polynomial_solver.h"
#include "polynomial_sampler.h"

using namespace mediation_layer;

int main(int argc, char** argv) {

  const size_t num_nodes = 10000;
  std::vector<double> times = {0};

  // Equality bounds paramater order is:
  // 1) Dimension index
  // 2) Node index
  // 3) Derivative index
  // 4) Value
  std::vector<EqualityBound> equality_bounds = {
    // The first node must constrain position, velocity, and acceleration
    EqualityBound(0,0,0,0),
    EqualityBound(0,0,1,0),
    EqualityBound(0,0,2,0),
  };

  for(size_t node_idx = 1; node_idx < num_nodes; ++node_idx) {
    times.push_back(node_idx);

    equality_bounds.push_back(EqualityBound(0,node_idx,0,node_idx));
  }

  const std::vector<LowerBound> lower_bounds = {};
  const std::vector<UpperBound> upper_bounds = {};

  PolynomialSolver::Options solver_options;
  solver_options.num_dimensions = 1;
  solver_options.polynomial_order = 7;
  solver_options.continuity_order = 4;
  solver_options.derivative_order = 4;
  solver_options.polish = true;       

  PolynomialSolver solver(solver_options);
  const PolynomialPath path
    = solver.Run(times, equality_bounds, upper_bounds, lower_bounds);

  return EXIT_SUCCESS;
}
