// Author: Tucker Haydon

#include <iostream>
#include <cstdlib>
#include <vector>

#include <Eigen/Dense>
#include <osqp.h>

#include "polynomial_solver.h"

namespace p4 {
  bool PolynomialSolver::Setup(
      const std::vector<double>& times,
      const std::vector<NodeEqualityBound>& explicit_node_equality_bounds,
      const std::vector<NodeInequalityBound>& explicit_node_inequality_bounds,
      const std::vector<SegmentInequalityBound>& explicit_segment_inequality_bounds) {

    // Checks
    if(times.size() < 2) {
      std::cerr << "PolynomialSolver::Setup -- Time vector must have a size greater than one." << std::endl;
      std::exit(EXIT_FAILURE);
    }

    // Cache user input in the workspace
    this->workspace_.times = times;
    this->workspace_.explicit_node_equality_bounds = explicit_node_equality_bounds;
    this->workspace_.explicit_node_inequality_bounds = explicit_node_inequality_bounds;
    this->workspace_.explicit_segment_inequality_bounds = explicit_segment_inequality_bounds;

    // Cache constants in the workspace
    this->workspace_.constants.num_dimensions = 
      this->options_.num_dimensions;
    this->workspace_.constants.polynomial_order = 
      this->options_.polynomial_order;
    this->workspace_.constants.derivative_order = 
      this->options_.derivative_order;
    this->workspace_.constants.continuity_order = 
      this->options_.continuity_order;
    this->workspace_.constants.num_intermediate_points = 
      this->options_.num_intermediate_points;
    this->workspace_.constants.num_nodes = 
      times.size();
    this->workspace_.constants.num_segments = 
      this->workspace_.constants.num_nodes - 1;
    this->workspace_.constants.num_params_per_node_per_dim = 
      this->workspace_.constants.polynomial_order + 1;
    this->workspace_.constants.num_params_per_segment_per_dim = 
      this->workspace_.constants.polynomial_order + 1;
    this->workspace_.constants.num_params_per_node = 
      this->workspace_.constants.num_dimensions * 
      this->workspace_.constants.num_params_per_node_per_dim;
    this->workspace_.constants.num_params_per_segment = 
      this->workspace_.constants.num_dimensions * 
      this->workspace_.constants.num_params_per_segment_per_dim;
    this->workspace_.constants.total_num_params = 
      this->workspace_.constants.num_params_per_node * 
      this->workspace_.constants.num_nodes;

    // Explicit constraints are provided
    // Add two to the number of intermediate points due to the implicit endpoint
    // constraints
    const size_t num_explicit_constraints = 0
      + this->workspace_.explicit_node_equality_bounds.size() 
      + this->workspace_.explicit_node_inequality_bounds.size() 
      + this->workspace_.explicit_segment_inequality_bounds.size() 
      * (this->workspace_.constants.num_intermediate_points+2);

    // Implicit constraints are continuity constraints
    const size_t num_implicit_constraints = 
      this->workspace_.constants.num_segments
      *(this->workspace_.constants.continuity_order+1)
      *this->workspace_.constants.num_dimensions;

    this->workspace_.constants.num_constraints = num_explicit_constraints + num_implicit_constraints;

    // Allocate constraints
    this->workspace_.lower_bound_vec.resize(this->workspace_.constants.num_constraints);
    this->workspace_.upper_bound_vec.resize(this->workspace_.constants.num_constraints);
    this->workspace_.sparse_constraint_mat = Eigen::SparseMatrix<double>(
        this->workspace_.constants.num_constraints, 
        this->workspace_.constants.total_num_params);

    // Fill constraints
    this->SetConstraints<double>(
        this->workspace_.times,
        this->workspace_.lower_bound_vec, 
        this->workspace_.upper_bound_vec, 
        this->workspace_.sparse_constraint_mat);

    // Allocate quadratic matrix
    this->workspace_.sparse_quadratic_mat = Eigen::SparseMatrix<double>(
        this->workspace_.constants.total_num_params, 
        this->workspace_.constants.total_num_params);

    // Fill quadratic matrix
    this->SetQuadraticCost<double>(this->workspace_.sparse_quadratic_mat);

    // Workspace is now set up
    this->workspace_.setup = true;
    return true;
  }


  PolynomialSolver::Solution PolynomialSolver::Run() {
    // Convert to solver data types
    csc* P = nullptr;
    csc* A = nullptr;

    Eigen2OSQP(this->workspace_.sparse_quadratic_mat, P);
    Eigen2OSQP(this->workspace_.sparse_constraint_mat, A);

    c_float q[this->workspace_.constants.total_num_params];
    for(size_t param_idx = 0; param_idx < this->workspace_.constants.total_num_params; ++param_idx) {
      q[param_idx] = 0;
    }

    c_float l[this->workspace_.constants.num_constraints], u[this->workspace_.constants.num_constraints];
    for(size_t row_idx = 0; row_idx < this->workspace_.constants.num_constraints; ++row_idx) {
      l[row_idx] = this->workspace_.lower_bound_vec(row_idx);
      u[row_idx] = this->workspace_.upper_bound_vec(row_idx);
    }

    // Run the solver
    // Allocate and populate data
    std::shared_ptr<OSQPData> data = std::shared_ptr<OSQPData>(
        (OSQPData *)c_malloc(sizeof(OSQPData)),
        [](OSQPData* data) {
          c_free(data->A);
          c_free(data->P);
        });
    data->n = this->workspace_.constants.total_num_params;
    data->m = this->workspace_.constants.num_constraints;
    data->P = P;
    data->q = q;
    data->A = A;
    data->l = l;
    data->u = u;

    // Allocate and prepare workspace
    // Workspace shared pointer requires custom destructor
    PolynomialSolver::Solution solution;
    solution.constants = this->workspace_.constants;
    solution.data      = data;
    solution.workspace = std::shared_ptr<OSQPWorkspace>(
        osqp_setup(data.get(), &this->options_.osqp_settings),
        [](OSQPWorkspace* workspace) { 
          osqp_cleanup(workspace);
        });

    // Solve
    osqp_solve(solution.workspace.get());

    // Return the solution
    return solution;
  }

  void PolynomialSolver::Options::Check() {
    if(this->num_dimensions < 1) {
      std::cerr << "PolynomialSolver::Options::Check -- Number of dimensions must be greater than zero." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }

  std::vector<std::vector<Eigen::VectorXd>> PolynomialSolver::Solution::Coefficients() const {
    std::vector<std::vector<Eigen::VectorXd>> coefficients;

    coefficients.resize(this->constants.num_dimensions);
    for(size_t dimension_idx = 0; dimension_idx < this->constants.num_dimensions; ++dimension_idx) {
      coefficients[dimension_idx].resize(this->constants.num_nodes);
      for(size_t node_idx = 0; node_idx < this->constants.num_nodes; ++node_idx) {
        coefficients[dimension_idx][node_idx].resize(this->constants.num_params_per_node_per_dim);
        for(size_t coefficient_idx = 0; coefficient_idx < this->constants.num_params_per_node_per_dim; ++coefficient_idx) {
          const size_t parameter_idx = 0
            // Get to the right dimension
            + this->constants.num_params_per_node_per_dim 
            * this->constants.num_nodes 
            * dimension_idx
            // Get to the right node
            + this->constants.num_params_per_node_per_dim 
            * node_idx
            // Get to the right parameter idx
            + coefficient_idx;

          coefficients[dimension_idx][node_idx](coefficient_idx)
            = this->workspace->solution->x[parameter_idx];
        }
      }
    }
    return coefficients;
  }

  Eigen::VectorXd PolynomialSolver::Solution::Coefficients(
      const size_t dimension_idx, 
      const size_t node_idx) const {

    Eigen::VectorXd coefficients;
    coefficients.resize(this->constants.num_params_per_node_per_dim);
    for(size_t coefficient_idx = 0; coefficient_idx < this->constants.num_params_per_node_per_dim; ++coefficient_idx) {
      const size_t parameter_idx = 0
        // Get to the right dimension
        + this->constants.num_params_per_node_per_dim 
        * this->constants.num_nodes 
        * dimension_idx
        // Get to the right node
        + this->constants.num_params_per_node_per_dim 
        * node_idx
        // Get to the right parameter idx
        + coefficient_idx;

      coefficients(coefficient_idx)
        = this->workspace->solution->x[parameter_idx];
    }

    return coefficients;
  }
}

