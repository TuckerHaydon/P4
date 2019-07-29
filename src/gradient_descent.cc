// Author: Tucker Haydon

#include "gradient_descent.h"

#include <memory>
#include <iostream>

namespace p4 {

  GradientDescent::Solution GradientDescent::Run(
      const std::vector<double>& initial_times) {
    // Setup and configure
    // Require polish, no scaling
    PolynomialSolver::Options solver_options = 
      this->options_.solver_options;
    solver_options.osqp_settings.scaling    = 0;
    solver_options.osqp_settings.polish     = 1;
    auto solver = std::make_shared<PolynomialSolver>(solver_options);

    // Iteration variables
    std::vector<double> times = initial_times;
    PolynomialSolver::Solution solver_solution = solver->Run(times);
    double cost = solver_solution.workspace->info->obj_val;
    bool solution_found = false;

    for(size_t iteration_idx = 0; iteration_idx < this->options_.max_iterations; ++iteration_idx) {
      // Determine the gradient
      const Gradient::Options gradient_options 
        = this->options_.gradient_options;
      Gradient gradient(gradient_options);
      Gradient::Solution gradient_solution = 
        gradient.Run(times, solver, solver_solution);

      // Perform line search along the gradient
      const LineSearch::Options line_search_options 
        = this->options_.line_search_options;
      const LineSearch line_search(line_search_options);
      const LineSearch::Solution line_search_solution 
        = line_search.Run(times, solver, solver_solution, gradient_solution);

      // Update the times
      times = line_search_solution.times;

      // Evaluate terminal conditions
      solver_solution = solver->Run(times);
      double new_cost = solver_solution.workspace->info->obj_val;
      if(cost - new_cost < this->options_.cost_threshhold) {
        solution_found = true;
        break;
      } else {
        cost = new_cost;
      }
    }

    Solution solution;
    solution.solver_solution = solver_solution;
    solution.times = times;
    solution.success = solution_found;

    return solution;
  }
}
