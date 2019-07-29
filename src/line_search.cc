// Author: Tucker Haydon

#include "line_search.h"
#include "gradient.h"
#include <iostream>

namespace p4 {
  LineSearch::Solution LineSearch::Run(
    const std::vector<double>& initial_times,
    const std::shared_ptr<PolynomialSolver>& solver) {

    // Prepare times Eigen vector
    Eigen::Matrix<double, Eigen::Dynamic, 1> times(initial_times.size());
    for(size_t time_idx = 0; time_idx < initial_times.size(); ++time_idx) {
      times(time_idx) = initial_times[time_idx];
    }

    // Initial search point.
    PolynomialSolver::Solution initial_solver_solution = solver->Run(initial_times);
    double initial_cost = initial_solver_solution.workspace->info->obj_val;

    // Determine gradient
    PolynomialGradient::Options gradient_options;
    PolynomialGradient gradient(gradient_options);
    PolynomialGradient::Solution gradient_solution = 
      gradient.Run(initial_times, solver, initial_solver_solution);
    
    // Line search
    double alpha = this->options_.alpha_0;
    double previous_cost = initial_cost;
    Eigen::Matrix<double, Eigen::Dynamic, 1> previous_times = times;
    for(size_t iteration_idx = 0; iteration_idx < this->options_.max_iterations; ++iteration_idx) {
      // Upate step
      Eigen::Matrix<double, Eigen::Dynamic, 1> candidate_times = 
        // TODO: Is this plus or minus?
        times - alpha * gradient_solution.gradient;

      // Check candidate_times valid
      // Must be monotonic
      bool valid = true;
      for(size_t time_idx = 0; time_idx < candidate_times.rows()-1; ++time_idx) {
        if(candidate_times(time_idx+1) - candidate_times(time_idx) < 0) {
          valid = false;
          break;
        }
      }

      if(false == valid) {
        break;
      }

      PolynomialSolver::Solution solver_solution = 
        solver->Run(std::vector<double>(candidate_times.data(), 
                                        candidate_times.data() + candidate_times.rows()));
      double candidate_cost = solver_solution.workspace->info->obj_val;

      if(candidate_cost < previous_cost) {
        previous_cost = candidate_cost;
        previous_times = candidate_times;
        alpha = 2.0 * alpha;
      } else {
        alpha = 0.5 * alpha;
      }
    }

    Solution solution;
    solution.times = std::vector<double>(
        previous_times.data(), 
        previous_times.data() + previous_times.rows());

    return solution;
  }

}
