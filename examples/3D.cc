// Author: Tucker Haydon

#include <cstdlib>

#include <Eigen/Dense>
#include "gnuplot-iostream.h"

#include "polynomial_solver.h"
#include "polynomial_sampler.h"

using namespace mediation_layer;

int main(int argc, char** argv) {

  const std::vector<double> times = {0, 1, 2.5, 3.3};

  // Equality bounds paramater order is:
  // 1) Dimension index
  // 2) Node index
  // 3) Derivative index
  // 4) Value
  const std::vector<NodeEqualityBound> equality_bounds = {
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

    // The fourth node constrains position
    NodeEqualityBound(0,3,0,0),
    NodeEqualityBound(1,3,0,0),
    NodeEqualityBound(2,3,0,1),
  };

  PolynomialSolver::Options solver_options;
  solver_options.num_dimensions = 3;     // 3D
  solver_options.polynomial_order = 8;   // Fit an 7th-order polynomial
  solver_options.continuity_order = 4;   // Require continuity to the 4th order
  solver_options.derivative_order = 4;   // Minimize the 4th order (snap)
  solver_options.polish = true;          // Polish the solution

  PolynomialSolver solver(solver_options);
  const PolynomialPath path
    = solver.Run(times, equality_bounds,{},{});

  // Sampling and Plotting
  { // Plot acceleration profiles
    PolynomialSampler::Options sampler_options;
    sampler_options.frequency = 100;
    sampler_options.derivative_order = 2;

    PolynomialSampler sampler(sampler_options);
    Eigen::MatrixXd samples = sampler.Run(times, path);

    std::vector<double> t_hist, x_hist, y_hist, z_hist;
    for(size_t time_idx = 0; time_idx < samples.cols(); ++time_idx) {
      t_hist.push_back(samples(0,time_idx));
      x_hist.push_back(samples(1,time_idx));
      y_hist.push_back(samples(2,time_idx));
      z_hist.push_back(samples(3,time_idx));
    }

    Gnuplot gp;
    gp << "plot '-' using 1:2 with lines title 'X-Acceleration'";
    gp << ", '-' using 1:2 with lines title 'Y-Acceleration'";
    gp << ", '-' using 1:2 with lines title 'Z-Acceleration'";
    gp << std::endl;
    gp.send1d(boost::make_tuple(t_hist, x_hist));
    gp.send1d(boost::make_tuple(t_hist, y_hist));
    gp.send1d(boost::make_tuple(t_hist, z_hist));
    gp << "set grid" << std::endl;
    gp << "replot" << std::endl;
  }

  { // Plot velocity profiles
    PolynomialSampler::Options sampler_options;
    sampler_options.frequency = 100;
    sampler_options.derivative_order = 1;

    PolynomialSampler sampler(sampler_options);
    Eigen::MatrixXd samples = sampler.Run(times, path);

    std::vector<double> t_hist, x_hist, y_hist, z_hist;
    for(size_t time_idx = 0; time_idx < samples.cols(); ++time_idx) {
      t_hist.push_back(samples(0,time_idx));
      x_hist.push_back(samples(1,time_idx));
      y_hist.push_back(samples(2,time_idx));
      z_hist.push_back(samples(3,time_idx));
    }

    Gnuplot gp;
    gp << "plot '-' using 1:2 with lines title 'X-Velocity'";
    gp << ", '-' using 1:2 with lines title 'Y-Velocity'";
    gp << ", '-' using 1:2 with lines title 'Z-Velocity'";
    gp << std::endl;
    gp.send1d(boost::make_tuple(t_hist, x_hist));
    gp.send1d(boost::make_tuple(t_hist, y_hist));
    gp.send1d(boost::make_tuple(t_hist, z_hist));
    gp << "set grid" << std::endl;
    gp << "replot" << std::endl;
  }

  { // Plot 3D position
    PolynomialSampler::Options sampler_options;
    sampler_options.frequency = 50;
    sampler_options.derivative_order = 0;

    PolynomialSampler sampler(sampler_options);
    Eigen::MatrixXd samples = sampler.Run(times, path);

    std::vector<double> t_hist, x_hist, y_hist, z_hist;
    for(size_t time_idx = 0; time_idx < samples.cols(); ++time_idx) {
      t_hist.push_back(samples(0,time_idx));
      x_hist.push_back(samples(1,time_idx));
      y_hist.push_back(samples(2,time_idx));
      z_hist.push_back(samples(3,time_idx));
    }

    Gnuplot gp;
    gp << "splot '-' using 1:2:3 with lines title 'Trajectory'" << std::endl;
    gp.send1d(boost::make_tuple(x_hist, y_hist, z_hist));
    gp << "set grid" << std::endl;
    gp << "replot" << std::endl;

    // Must keep position gp in scope to rotate 3D graph
    std::cout << "Press enter to exit." << std::endl;
    std::cin.get();
  }





  return EXIT_SUCCESS;
}
