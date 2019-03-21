// Author: Tucker Haydon

#include <cstdlib>

#include <Eigen/Dense>
#include "gnuplot-iostream.h"

#include "polynomial_solver.h"
#include "polynomial_sampler.h"

using namespace mediation_layer;

int main(int argc, char** argv) {

  const std::vector<double> times = {0, 0.5, 1, 1.5};

  // Equality bounds paramater order is:
  // 1) Dimension index
  // 2) Node index
  // 3) Derivative index
  // 4) Value
  const std::vector<NodeEqualityBound> node_equality_bounds = {
    // The first node must constrain position, velocity, and acceleration
    NodeEqualityBound(0,0,0,0),
    NodeEqualityBound(1,0,0,0),
    NodeEqualityBound(2,0,0,0),
    NodeEqualityBound(0,0,1,0),
    NodeEqualityBound(1,0,1,0),
    NodeEqualityBound(2,0,1,0),
    NodeEqualityBound(0,0,2,0),
    NodeEqualityBound(1,0,2,0),
    NodeEqualityBound(2,0,2,0),

    // The second node constrains position
    NodeEqualityBound(0,1,0,1),
    NodeEqualityBound(1,1,0,0),
    NodeEqualityBound(2,1,0,0),

    // The third node constrains position
    NodeEqualityBound(0,2,0,1),
    NodeEqualityBound(1,2,0,1),
    NodeEqualityBound(2,2,0,1),
  };

  const std::vector<NodeInequalityBound> node_inequality_bounds = {
    // The fourth node constrain position
    NodeInequalityBound(0,3,0,0,0),
    NodeInequalityBound(1,3,0,0,0),
    NodeInequalityBound(2,3,0,-0.2,0.2),
  };

  // Order: segment, derivative, mapping, value
  const std::vector<SegmentInequalityBound> segment_inequality_bounds = {
    // Constrain y pos to corridor
    SegmentInequalityBound(0,0,Eigen::Vector3d(0,1,0), 0.001),
    SegmentInequalityBound(0,0,Eigen::Vector3d(0,-1,0),0.001),

    // Constrain z pos to corridor
    SegmentInequalityBound(0,0,Eigen::Vector3d(0,0,1),0.001),
    SegmentInequalityBound(0,0,Eigen::Vector3d(0,0,-1),0.001),

    // Constrain x vel below 3 m/s
    SegmentInequalityBound(0,1,Eigen::Vector3d(1,0,0),3),
  };

  PolynomialSolver::Options solver_options;
  solver_options.num_dimensions = 3;     // 3D
  solver_options.polynomial_order = 7;   // Fit an 7th-order polynomial
  solver_options.continuity_order = 3;   // Require continuity to the 4th order
  solver_options.derivative_order = 2;   // Minimize the 4th order (snap)
  solver_options.polish = true;          // Polish the solution

  PolynomialSolver solver(solver_options);
  const PolynomialPath path
    = solver.Run(
        times, 
        node_equality_bounds, 
        node_inequality_bounds, 
        segment_inequality_bounds);

  PolynomialSampler::Options sampler_options;
  sampler_options.frequency = 20;

  PolynomialSampler sampler(sampler_options);
  Eigen::MatrixXd samples = sampler.Run(times, path);

  // Plotting
  std::vector<double> t_hist, x_hist, y_hist, z_hist;
  for(size_t time_idx = 0; time_idx < samples.cols(); ++time_idx) {
    t_hist.push_back(samples(0,time_idx));
    x_hist.push_back(samples(1,time_idx));
    y_hist.push_back(samples(2,time_idx));
    z_hist.push_back(samples(3,time_idx));
  }

  Gnuplot gp;
  gp << "splot '-' using 1:2:3 with lines" << std::endl;
  gp.send1d(boost::make_tuple(x_hist, y_hist, z_hist));
  gp << "set grid" << std::endl;

  std::cout << "Press enter to exit." << std::endl;
  std::cin.get();


  return EXIT_SUCCESS;
}
