// Author: Tucker Haydon

#include "polynomial_sampler.h"
#include "common.h"

namespace p4 {
  // Sample a polynomial path solution
  Eigen::MatrixXd PolynomialSampler::Run(
      const std::vector<double>& times,
      const PolynomialSolver::Solution& solution) {

    // Reshape solution into usable data structure
    std::vector<std::vector<Eigen::VectorXd>> coefficients =
      solution.Coefficients();

    // Helper constants
    const size_t num_dimensions = solution.constants.num_dimensions;
    const size_t num_nodes = solution.constants.num_nodes;
    const size_t polynomial_order = solution.constants.polynomial_order;
    const size_t num_samples 
      = static_cast<size_t>((times.back() - times.front()) * this->options_.frequency);

    // Prepare the sample matrix
    Eigen::MatrixXd samples;
    samples.resize(num_dimensions + 1, num_samples);
    samples.fill(0);

    size_t node_idx = 0;
    size_t sample_idx = 0;
    double start_time = times.front();

    for(size_t sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
      const double path_time = start_time + sample_idx / this->options_.frequency;

      if(path_time > times[node_idx + 1]) {
        node_idx++;
      }

      // Push time
      samples(0, sample_idx) = path_time;

      for(size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
        const double alpha = times[node_idx+1] - times[node_idx];
        const double t = path_time - times[node_idx];
        const double tau = t / alpha;

        const Eigen::VectorXd polynomial_coefficients 
          = coefficients[dimension_idx][node_idx];
        const Eigen::MatrixXd tau_vec
          = TimeVector(polynomial_order, this->options_.derivative_order, tau);

        // Time is the first dimension. Shift the index down.
        samples(dimension_idx + 1, sample_idx) 
          = (polynomial_coefficients.transpose() * tau_vec)(0,0) 
          / std::pow(alpha, this->options_.derivative_order);
      }
    }
    return samples;
  }
}
