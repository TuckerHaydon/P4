// Author: Tucker Haydon

#include <iostream>
#include <cstdlib>

#include "solver.h"

namespace mediation_layer {
  template <
    size_t dimension, 
    size_t polynomial_order, 
    size_t derivative_order, 
    size_t continuity_order>
  PolynomialSolver::Solution Run(
      const std::vector<double>& times,
      const std::vector<EqualityBound>& equality_bounds,
      const std::vector<UpperBound>& upper_bounds,
      const std::vector<LowerBound>& lower_bounds) {

    if(dimension < 1) {
      std::cerr << "PolynomialSolver::Run -- Dimension must be positive." << std::endl;
      std::exit(EXIT_FAILURE);
    }

    if(polynomial_order < derivative_order + 3) {
      std::cerr << "PolynomialSolver::Run -- Polynomial order must be 3 or more orders greater than derivtive order." << std::endl;
      std::exit(EXIT_FAILURE);
    }

    if(polynomial_order <= continuity_order) {
      std::cerr << "PolynomialSolver::Run -- Polynomial order must be greated than continuity order" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    if(times.size() < 2) {
      std::cerr << "PolynomialSolver::Run -- Too few times specified." << std::endl;
      std::exit(EXIT_FAILURE);
    }


    const size_t num_nodes = times.size();
    const size_t num_segments = num_nodes - 1;

    /* The number of constraints is governed by the following rules:
     * 1) The first node must be constrained by at least the 0th, 1st, and 2nd
     *    derivatives: 
     *      3*dimension constraints
     * 2) Every other node must be constrained by at least the 0th derivative: 
     *      1*(num_nodes - 1)*dimension constraints
     * 3) The end of every segment must match the beginning of the next segment.
     *    After propagating the initial node through the segment polynomial, the
     *    resulting node should match the initial node of the next segment for
     *    continuity_order derivatives.
     *      num_segments*(continuity_order+1)*dimension constraints
     *  4) Corridor Constraints: TODO
     *      
     */
    const size_t min_num_constraints = 0
      + 3*dimension 
      + (num_nodes - 1)*dimension 
      + num_segments*(continuity_order+1)*dimension;

    // Explicit constraints are provided
    const size_t num_explicit_constraints = equality_bounds.size() + upper_bounds.size() + lower_bounds.size();

    // Implicit constraints are continuity constraints
    const size_t num_implicit_constraints = num_segments*(continuity_order+1)*dimension;

    const size_t num_params_per_node_per_dim = polynomial_order + 1;
    const size_t num_params_per_segment_per_dim = polynomial_order + 1;
    const size_t num_params_per_node = dimension * num_params_per_node_per_dim;
    const size_t num_params_per_segment = dimension * num_params_per_segment_per_dim;
    const size_t total_num_params = num_params_per_node * num_nodes;
  }

};
