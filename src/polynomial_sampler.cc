// Author: Tucker Haydon

#include "polynomial_sampler.h"

namespace p4 {
  namespace {
    // Computes n!
    size_t factorial(size_t n) {
      return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
    }

    // Constructs and takes the derivative of a vector of the form:
    //   [ (1/0! dt^0), (1/1! dt^1), (1/2! dt^2) ... ]'
    // The derivative can be efficiently easily calculated by prepadding zeros
    // to the front of the vector and shifting to the right. This follows from
    // from the structure of the time vector.
    //
    // See the theory documentation for further details.
    Eigen::MatrixXd TimeVector(
        const size_t polynomial_order, 
        const size_t derivative_order, 
        const double time) {
      Eigen::MatrixXd base_coefficient_vec;
      base_coefficient_vec.resize(polynomial_order + 1,1);
      for(size_t idx = 0; idx < polynomial_order + 1; ++idx) {
        // std::pow(0,0) undefined. Define as 1.0.
        if(0.0 == time && 0 == idx) {
          base_coefficient_vec(idx, 0) = 1.0 / factorial(idx);
        } else {
          base_coefficient_vec(idx, 0) = std::pow(time, idx) / factorial(idx);
        }
      }
    
      Eigen::MatrixXd ones_vec;
      ones_vec.resize(polynomial_order + 1 - derivative_order, 1);
      ones_vec.fill(1);
    
      Eigen::MatrixXd shift_mat;
      shift_mat.resize(polynomial_order + 1, polynomial_order + 1);
      shift_mat.fill(0);
      shift_mat.diagonal(-1*derivative_order) = ones_vec;
    
      Eigen::MatrixXd coefficient_vec;
      coefficient_vec.resize(polynomial_order + 1, 1);
      coefficient_vec = shift_mat * base_coefficient_vec;
    
      return coefficient_vec;
    }
  }

  // Sample a polynomial path solution
  Eigen::MatrixXd PolynomialSampler::Run(
      const std::vector<double>& times,
      const PolynomialSolver::Solution& solution) {

    // Reshape solution into usable data structure
    std::vector<std::vector<Eigen::VectorXd>> coefficients =
      solution.Coefficients();

    // Helper constants
    const size_t num_dimensions = solution.num_dimensions;
    const size_t num_nodes = solution.num_nodes;
    const size_t polynomial_order = solution.polynomial_order;
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
