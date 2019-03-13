// Author: Tucker Haydon

#include "polynomial_sampler.h"

namespace mediation_layer {
  namespace {
    size_t factorial(size_t n) {
      return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
    }

    Eigen::MatrixXd TimeVector(
        const size_t polynomial_order, 
        const size_t derivative_order, 
        const double time) {
      Eigen::MatrixXd base_coefficient_vec;
      base_coefficient_vec.resize(polynomial_order + 1,1);
      for(size_t idx = 0; idx < polynomial_order + 1; ++idx) {
        base_coefficient_vec(idx, 0) = std::pow(time, idx) / factorial(idx);
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

    Eigen::MatrixXd ScaleMatrix(
        const size_t polynomial_order,
        const double alpha) {
      Eigen::MatrixXd scale_mat;
      scale_mat.resize(polynomial_order + 1, polynomial_order + 1);
      scale_mat.fill(0);

      for(size_t polynomial_idx = 0; polynomial_idx < polynomial_order + 1; ++polynomial_idx) {
        scale_mat(polynomial_idx, polynomial_idx) = std::pow(alpha, polynomial_idx);
      }

      return scale_mat;
    }
  }

  Eigen::MatrixXd PolynomialSampler::Run(
      const std::vector<double>& times,
      const PolynomialPath& path) {

    const size_t num_dimensions = path.coefficients.size();
    const size_t num_nodes = path.coefficients[0].cols();
    const size_t polynomial_order = path.coefficients[0].rows() - 1;
    const size_t num_samples 
      = static_cast<size_t>((times.back() - times.front()) * this->options_.frequency);

    // Prepare the sample matrix
    Eigen::MatrixXd samples;
    samples.resize(num_dimensions, num_samples);
    samples.fill(0);

    size_t node_idx = 0;
    size_t sample_idx = 0;
    double start_time = times.front();

    for(size_t sample_idx = 0; sample_idx < num_samples; ++sample_idx) {
      const double time = start_time + sample_idx * this->options_.frequency;

      if(time > times[node_idx + 1]) {
        node_idx++;
      }

      for(size_t dimension_idx = 0; dimension_idx < num_dimensions; ++dimension_idx) {
        const double alpha = 1.0 / (times[node_idx+1] - times[node_idx]);

        const Eigen::MatrixXd polynomial_coefficients 
          = path.coefficients[dimension_idx].col(node_idx);
        const Eigen::MatrixXd scale_mat = ScaleMatrix(polynomial_order, alpha);
        const Eigen::MatrixXd time_vector = TimeVector(polynomial_order, 0, time - times[node_idx]);

        samples(dimension_idx, sample_idx) = (polynomial_coefficients * scale_mat * time_vector)(0,0);
      }
    }

    return samples;

  }

}
