// Author: Tucker Haydon

#include <cstdlib>
#include <Eigen/Dense>

#include "polynomial_solver.h"

using namespace mediation_layer;

void test_OneDimensionTwoNode() {
  PolynomialSolver::Options options;
  options.num_dimensions = 1;
  options.polynomial_order = 5;
  options.continuity_order = 2;
  options.derivative_order = 2;
  options.polish = true;

  PolynomialSolver solver(options);

  const std::vector<double> times = {0, 1};

  // PVA(0,0,0) -- PVA(1,~,~)
  const std::vector<EqualityBound> equality_bounds = {
    EqualityBound(0,0,0,0),
    EqualityBound(0,0,1,0),
    EqualityBound(0,0,2,0),

    EqualityBound(0,1,0,1)
  };

  const std::vector<LowerBound> lower_bounds = {};
  const std::vector<UpperBound> upper_bounds = {};

  const PolynomialSolver::Solution solution 
    = solver.Run(times, equality_bounds, upper_bounds, lower_bounds);
}

void test_TwoDimensionTwoNode() {
  PolynomialSolver::Options options;
  options.num_dimensions = 2;
  options.polynomial_order = 5;
  options.continuity_order = 2;
  options.derivative_order = 2;

  PolynomialSolver solver(options);

  const std::vector<double> times = {0, 1};

  const std::vector<EqualityBound> equality_bounds = {
    EqualityBound(0,0,0,1),
    EqualityBound(0,0,1,2),
    EqualityBound(0,0,2,3),
    EqualityBound(0,1,0,4),

    EqualityBound(1,0,0,5),
    EqualityBound(1,0,1,6),
    EqualityBound(1,0,2,7),
    EqualityBound(1,1,0,8),
  };

  const std::vector<LowerBound> lower_bounds = {};
  const std::vector<UpperBound> upper_bounds = {};

  const PolynomialSolver::Solution solution 
    = solver.Run(times, equality_bounds, upper_bounds, lower_bounds);
}

int main(int argc, char** argv) {
  test_OneDimensionTwoNode();
  // test_TwoDimensionTwoNode();

  return EXIT_SUCCESS;
}
