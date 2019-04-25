// Author: Tucker Haydon

#include <cstdlib>

#include <Eigen/Dense>

#include "polynomial_solver.h"
#include "polynomial_sampler.h"

using namespace p4;

int main(int argc, char** argv) {

  const size_t num_nodes = 10000;
  std::vector<double> times = {0};

  // Equality bounds paramater order is:
  // 1) Dimension index
  // 2) Node index
  // 3) Derivative index
  // 4) Value
  std::vector<NodeEqualityBound> equality_bounds = {
    // The first node must constrain position, velocity, and acceleration
    NodeEqualityBound(0,0,0,0),
    NodeEqualityBound(0,0,1,0),
    NodeEqualityBound(0,0,2,0),
  };

  for(size_t node_idx = 1; node_idx < num_nodes; ++node_idx) {
    times.push_back(node_idx);

    equality_bounds.push_back(NodeEqualityBound(0,node_idx,0,node_idx));
  }

  PolynomialSolver::Options solver_options;
  solver_options.num_dimensions = 1;
  solver_options.polynomial_order = 7;
  solver_options.continuity_order = 4;
  solver_options.derivative_order = 4;
  solver_options.polish = true;       

  PolynomialSolver solver(solver_options);
  const PolynomialPath path
    = solver.Run(times, equality_bounds,{},{});

  return EXIT_SUCCESS;
}
