// Author: Tucker Haydon

#include <iostream>
#include <cstdlib>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <osqp.h>

#include "polynomial_solver.h"

namespace p4 {
  namespace {
    struct Info {   
      size_t num_dimensions;
      size_t polynomial_order;
      size_t derivative_order;
      size_t continuity_order;
      size_t num_intermediate_points;
      size_t num_nodes;
      size_t num_segments;
      size_t num_params_per_node_per_dim;
      size_t num_params_per_segment_per_dim;
      size_t num_params_per_node;
      size_t num_params_per_segment;
      size_t total_num_params;
      size_t num_constraints;
    };

    size_t factorial(size_t n) {
      return (n == 1 || n == 0) ? 1 : factorial(n - 1) * n;
    }

    Eigen::MatrixXd TimeVector(
        const size_t polynomial_order, 
        const size_t derivative_order, 
        const double dt) {
      Eigen::MatrixXd base_coefficient_vec;
      base_coefficient_vec.resize(polynomial_order + 1,1);
      for(size_t idx = 0; idx < polynomial_order + 1; ++idx) {
        // pow(0,0) is undefined. Define as 1.0.
        if(0.0 == dt && 0 == idx) {
          base_coefficient_vec(idx, 0) = 1.0 / factorial(idx);
        } else {
          base_coefficient_vec(idx, 0) = std::pow(dt, idx) / factorial(idx);
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

    // Generates a square matrix that is the integrated form of p(x)'p(x)
    Eigen::MatrixXd QuadraticMatrix(
        const size_t polynomial_order,
        const size_t derivative_order,
        const double dt) {
      Eigen::MatrixXd base_integrated_quadratic_matrix;
      base_integrated_quadratic_matrix.resize(polynomial_order + 1, polynomial_order + 1);
      base_integrated_quadratic_matrix.fill(0);
      for(size_t row = 0; row < polynomial_order + 1; ++row) {
        for(size_t col = 0; col < polynomial_order + 1; ++col) {
          base_integrated_quadratic_matrix(row, col) = 
            std::pow(dt, row + col + 1) 
            / factorial(row) 
            / factorial(col) 
            / (row + col + 1);
        }
      }
    
      // Vector of ones
      Eigen::MatrixXd ones_vec;
      ones_vec.resize(polynomial_order + 1 - derivative_order, 1);
      ones_vec.fill(1);
    
      // Shift the matrix down rows
      Eigen::MatrixXd row_shift_mat;
      row_shift_mat.resize(polynomial_order + 1, polynomial_order + 1);
      row_shift_mat.fill(0);
      row_shift_mat.diagonal(-1*derivative_order) = ones_vec;
    
      // Shift the matrix right cols
      Eigen::MatrixXd col_shift_mat;
      col_shift_mat.resize(polynomial_order + 1, polynomial_order + 1);
      col_shift_mat.fill(0);
      col_shift_mat.diagonal(+1*derivative_order) = ones_vec;
    
      Eigen::MatrixXd integrated_quadratic_matrix;
      integrated_quadratic_matrix.resize(polynomial_order + 1, polynomial_order + 1);
      integrated_quadratic_matrix = row_shift_mat * base_integrated_quadratic_matrix * col_shift_mat;
    
      return integrated_quadratic_matrix;
    }

    // Sets the upper and lower bound vectors for the equality and continuity
    // constraints.
    void SetConstraints(
        const Info& info,
        const std::vector<double>& times,
        const std::vector<NodeEqualityBound>& explicit_node_equality_bounds, 
        const std::vector<NodeInequalityBound>& explicit_node_inequality_bounds,
        const std::vector<SegmentInequalityBound>& explicit_segment_inequality_bounds,
        Eigen::MatrixXd& lower_bound_vec, 
        Eigen::MatrixXd& upper_bound_vec,
        std::vector<Eigen::Triplet<double>>& constraint_triplets
        ) {
      size_t constraint_idx = 0;
      for(size_t dimension_idx = 0; dimension_idx < info.num_dimensions; ++dimension_idx) { 
        for(size_t node_idx = 0; node_idx < info.num_nodes; ++node_idx) {
          for(size_t derivative_idx = 0; derivative_idx < info.num_params_per_segment_per_dim; ++derivative_idx) {

            // Equality Constraints
            for(const NodeEqualityBound& bound: explicit_node_equality_bounds) {
              if(
                  false == (bound.node_idx == node_idx) ||
                  false == (bound.dimension_idx == dimension_idx) || 
                  false == (bound.derivative_idx == derivative_idx)) {
                continue;
              }
              else {
                const double alpha = node_idx+1 < info.num_nodes ? (times[node_idx + 1] - times[node_idx]) : 1;

                // Bounds. Scaled by alpha. See documentation.
                lower_bound_vec(constraint_idx,0) = bound.value * std::pow(alpha, derivative_idx);
                upper_bound_vec(constraint_idx,0) = bound.value * std::pow(alpha, derivative_idx);

                // Constraints
                size_t parameter_idx = 0 
                  + derivative_idx 
                  + info.num_params_per_node_per_dim * node_idx
                  + info.num_params_per_node_per_dim * info.num_nodes * dimension_idx;
                constraint_triplets.emplace_back(constraint_idx, parameter_idx, 1);

                constraint_idx++;
              }
            }

            // Node inequality bound constraints
            for(const NodeInequalityBound& bound: explicit_node_inequality_bounds) {
              if(
                  false == (bound.node_idx == node_idx) ||
                  false == (bound.dimension_idx == dimension_idx) || 
                  false == (bound.derivative_idx == derivative_idx)) {
                continue;
              }
              else {
                const double alpha = node_idx+1 < info.num_nodes ? (times[node_idx + 1] - times[node_idx]) : 1;

                // Bounds. Scaled by alpha. See documentation.
                lower_bound_vec(constraint_idx,0) = bound.lower * std::pow(alpha, derivative_idx);
                upper_bound_vec(constraint_idx,0) = bound.upper * std::pow(alpha, derivative_idx);

                // Constraints
                size_t parameter_idx = 0 
                  + derivative_idx 
                  + info.num_params_per_node_per_dim * node_idx
                  + info.num_params_per_node_per_dim * info.num_nodes * dimension_idx;
                constraint_triplets.emplace_back(constraint_idx, parameter_idx, 1);

                constraint_idx++;
              }
            }
          }

          // Continuity constraints
          if(node_idx < info.num_segments) {
            const size_t num_continuity_constraints = info.continuity_order + 1;
            constexpr double delta_t = 1.0;
            const double alpha_k = times[node_idx + 1] - times[node_idx];
            const double alpha_kp1 = node_idx + 2 < info.num_nodes ? times[node_idx + 2] - times[node_idx + 1] : 1.0;


            for(size_t continuity_idx = 0; continuity_idx < num_continuity_constraints; ++continuity_idx) {
              // Bounds
              lower_bound_vec(constraint_idx,0) = 0;
              upper_bound_vec(constraint_idx,0) = 0;

              // Constraints. Scaled by alpha. See documentation.
              // Propagate the current node
              Eigen::MatrixXd segment_propagation_coefficients;
              segment_propagation_coefficients.resize(1, info.num_params_per_segment_per_dim);
              segment_propagation_coefficients.fill(0);
              segment_propagation_coefficients 
                = TimeVector(info.polynomial_order, continuity_idx, delta_t).transpose()
                / std::pow(alpha_k, continuity_idx);

              // Minus the next node
              Eigen::MatrixXd segment_terminal_coefficients;
              segment_terminal_coefficients.resize(1, info.num_params_per_segment_per_dim);
              segment_terminal_coefficients.fill(0);
              segment_terminal_coefficients(0,continuity_idx) = -1 / std::pow(alpha_kp1, continuity_idx);

              size_t current_segment_idx = 0 
                // Get to the right dimension
                + info.num_params_per_node_per_dim * info.num_nodes * dimension_idx
                // Get to the right node
                + info.num_params_per_node_per_dim * node_idx;
              size_t next_segment_idx = 0
                // Get to the right dimension
                + info.num_params_per_node_per_dim * info.num_nodes * dimension_idx
                // Get to the right node
                + info.num_params_per_node_per_dim * (node_idx + 1);

              for(size_t param_idx = 0; param_idx < info.num_params_per_segment_per_dim; ++param_idx) {
                constraint_triplets.emplace_back(
                    constraint_idx, 
                    current_segment_idx + param_idx, 
                    segment_propagation_coefficients(0, param_idx));
                // TODO: Just insert one terminal constraint
                constraint_triplets.emplace_back(
                    constraint_idx, 
                    next_segment_idx + param_idx, 
                    segment_terminal_coefficients(0, param_idx));
              }

              constraint_idx++;
            }
          }
        }
      }

      // Start and endpoints are not included in segment bounds
      const double dt = 1.0 / (info.num_intermediate_points + 2);

      // Segment lower bound constraints
      for(const SegmentInequalityBound& bound: explicit_segment_inequality_bounds) {
        const double alpha = times[bound.segment_idx+1] - times[bound.segment_idx];

        // point_idx == intermediate_point_idx
        for(size_t point_idx = 0; point_idx < info.num_intermediate_points; ++point_idx)  {
          // Bounds
          lower_bound_vec(constraint_idx,0) = -SegmentInequalityBound::INFTY;
          upper_bound_vec(constraint_idx,0) = bound.value * std::pow(alpha, bound.derivative_idx);

          // Start point not included
          double time = (1 + point_idx) * dt;

          Eigen::MatrixXd segment_propagation_coefficients;
          segment_propagation_coefficients.resize(1, info.num_params_per_segment_per_dim);
          segment_propagation_coefficients.fill(0);
          segment_propagation_coefficients 
            = TimeVector(info.polynomial_order, bound.derivative_idx, time).transpose();

          for(size_t dimension_idx = 0; dimension_idx < info.num_dimensions; ++dimension_idx) {
            Eigen::MatrixXd transform_coefficients;
            transform_coefficients.resize(1, info.num_params_per_segment_per_dim);
            transform_coefficients.fill(0);
            transform_coefficients = bound.mapping(dimension_idx, 0) 
              * segment_propagation_coefficients;

            size_t current_segment_idx = 0 
              // Get to the right dimension
              + info.num_params_per_node_per_dim * info.num_nodes * dimension_idx
              // Get to the right node
              + info.num_params_per_segment_per_dim * bound.segment_idx;

            for(size_t param_idx = 0; param_idx < info.num_params_per_segment_per_dim; ++param_idx) {
              constraint_triplets.emplace_back(
                  constraint_idx, 
                  current_segment_idx + param_idx, 
                  transform_coefficients(0, param_idx));
            }
          }

          constraint_idx++;
        }
      }
    }

    void SetQuadraticCost(
        const Info& info,
        std::vector<Eigen::Triplet<double>>& quadratic_triplets) {
      const double delta_t = 1.0;
      const Eigen::MatrixXd quadratic_matrix = QuadraticMatrix(info.polynomial_order, info.derivative_order, delta_t);

      for(size_t dimension_idx = 0; dimension_idx < info.num_dimensions; ++dimension_idx) {
        // No cost for final node
        for(size_t node_idx = 0; node_idx < info.num_nodes - 1; ++node_idx) {
          const size_t parameter_idx = 0
            // Get to the right dimension
            + info.num_params_per_node_per_dim * info.num_nodes * dimension_idx
            // Get to the right node
            + info.num_params_per_node_per_dim * node_idx;
          for(size_t row = 0; row < info.num_params_per_node_per_dim; ++row) {
            for(size_t col = 0; col < info.num_params_per_node_per_dim; ++col) { 
              quadratic_triplets.emplace_back(
                  row + parameter_idx, 
                  col + parameter_idx, 
                  quadratic_matrix(row,col)
                  );
            }
          }
        }
      }
    }

    // Converts an en eigen sparse matrix into an OSQP sparse matrix
    // Reference: https://github.com/robotology/osqp-eigen
    void Eigen2OSQP(
        const Eigen::SparseMatrix<double> eigen_sparse_mat,
        csc*& osqp_mat) {

      // get number of row, columns and nonZeros from Eigen SparseMatrix
      c_int rows   = eigen_sparse_mat.rows();
      c_int cols   = eigen_sparse_mat.cols();
      c_int num_nz = eigen_sparse_mat.nonZeros();
    
      // get inner and outer index
      const int* innerIndexPtr    = eigen_sparse_mat.innerIndexPtr();
      const int* outerIndexPtr    = eigen_sparse_mat.outerIndexPtr();
      const int* innerNonZerosPtr = eigen_sparse_mat.innerNonZeroPtr();
    
      // get nonzero values
      const double* valuePtr = eigen_sparse_mat.valuePtr();
    
      // Allocate memory for csc matrix
      if(osqp_mat != nullptr){
        std::cerr << "osqp_mat pointer is not a null pointer! " << std::endl;
        std::exit(EXIT_FAILURE);
      }
    
      osqp_mat = csc_spalloc(rows, cols, num_nz, 1, 0);
    
      int innerOsqpPosition = 0;
      for(int k = 0; k < cols; ++k) {
          if (eigen_sparse_mat.isCompressed()) {
              osqp_mat->p[k] = static_cast<c_int>(outerIndexPtr[k]);
          } else {
              if (k == 0) {
                  osqp_mat->p[k] = 0;
              } else {
                  osqp_mat->p[k] = osqp_mat->p[k-1] + innerNonZerosPtr[k-1];
              }
          }
          for (typename Eigen::SparseMatrix<double>::InnerIterator it(eigen_sparse_mat,k); it; ++it) {
              osqp_mat->i[innerOsqpPosition] = static_cast<c_int>(it.row());
              osqp_mat->x[innerOsqpPosition] = static_cast<c_float>(it.value());
              innerOsqpPosition++;
          }
      }
      osqp_mat->p[static_cast<int>(cols)] = static_cast<c_int>(innerOsqpPosition);
    }
  }


  PolynomialPath PolynomialSolver::Run(
      const std::vector<double>& times,
      const std::vector<NodeEqualityBound>& explicit_node_equality_bounds,
      const std::vector<NodeInequalityBound>& explicit_node_inequality_bounds,
      const std::vector<SegmentInequalityBound>& explicit_segment_inequality_bounds) {

    this->options_.Check();

    if(times.size() < 2) {
      std::cerr << "PolynomialSolver::Run -- Too few times specified." << std::endl;
      std::exit(EXIT_FAILURE);
    }

    Info info;
    info.num_dimensions = this->options_.num_dimensions;
    info.polynomial_order = this->options_.polynomial_order;
    info.derivative_order = this->options_.derivative_order;
    info.continuity_order = this->options_.continuity_order;
    info.num_intermediate_points = this->options_.num_intermediate_points;
    info.num_nodes = times.size();
    info.num_segments = info.num_nodes - 1;
    info.num_params_per_node_per_dim = info.polynomial_order + 1;
    info.num_params_per_segment_per_dim = info.polynomial_order + 1;
    info.num_params_per_node = info.num_dimensions * info.num_params_per_node_per_dim;
    info.num_params_per_segment = info.num_dimensions * info.num_params_per_segment_per_dim;
    info.total_num_params = info.num_params_per_node * info.num_nodes;

    // Explicit constraints are provided
    const size_t num_explicit_constraints = 0
      + explicit_node_equality_bounds.size() 
      + explicit_node_inequality_bounds.size() 
      + explicit_segment_inequality_bounds.size() * info.num_intermediate_points;

    // Implicit constraints are continuity constraints
    const size_t num_implicit_constraints = info.num_segments*(info.continuity_order+1)*info.num_dimensions;

    info.num_constraints = num_explicit_constraints + num_implicit_constraints;

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
      + 3*info.num_dimensions 
      + (info.num_nodes - 1)*info.num_dimensions
      + info.num_segments*(info.continuity_order+1)*info.num_dimensions;

    if(info.num_constraints < min_num_constraints) {
      std::cerr << "PolynomialSolver::Run -- Too few constraints." << std::endl;
      std::exit(EXIT_FAILURE);
    }

    /*
     * CONSTRAINTS
     */
    Eigen::MatrixXd lower_bound_vec, upper_bound_vec;
    lower_bound_vec.resize(info.num_constraints, 1);
    upper_bound_vec.resize(info.num_constraints, 1);

    std::vector<Eigen::Triplet<double>> constraint_triplets;
    SetConstraints(
        info, 
        times,
        explicit_node_equality_bounds,
        explicit_node_inequality_bounds,
        explicit_segment_inequality_bounds,
        lower_bound_vec, 
        upper_bound_vec, 
        constraint_triplets);

    // Triplets to sparse mat
    Eigen::SparseMatrix<double> sparse_constraint_mat(
        info.num_constraints, 
        info.total_num_params);
    sparse_constraint_mat.setFromTriplets(
        constraint_triplets.begin(), 
        constraint_triplets.end());

    /*
     * QUADRATIC MATRIX
     */
    std::vector<Eigen::Triplet<double>> quadratic_triplets;
    SetQuadraticCost(info, quadratic_triplets);

    // Triplets to sparse mat
    Eigen::SparseMatrix<double> sparse_quadratic_mat(
        info.total_num_params, 
        info.total_num_params);
    sparse_quadratic_mat.setFromTriplets(
        quadratic_triplets.begin(), 
        quadratic_triplets.end());

    /*
     * CONVERT EIGEN TO OSQP
     */
    csc* P = nullptr;
    csc* A = nullptr;

    Eigen2OSQP(sparse_quadratic_mat, P);
    Eigen2OSQP(sparse_constraint_mat, A);

    c_float q[info.total_num_params];
    for(size_t param_idx = 0; param_idx < info.total_num_params; ++param_idx) {
      q[param_idx] = 0;
    }

    c_float l[info.num_constraints], u[info.num_constraints];
    for(size_t row_idx = 0; row_idx < info.num_constraints; ++row_idx) {
      l[row_idx] = lower_bound_vec(row_idx, 0);
      u[row_idx] = upper_bound_vec(row_idx, 0);
    }

    /*
     * RUN THE SOLVER
     */
    // Reference: https://osqp.org/docs/examples/demo.html
    // Problem settings
    OSQPSettings* settings = (OSQPSettings *)c_malloc(sizeof(OSQPSettings));

    // Structures
    OSQPWorkspace* work;
    OSQPData* data;  

    // Populate data
    data = (OSQPData*)c_malloc(sizeof(OSQPData));
    data->n = info.total_num_params;
    data->m = info.num_constraints;
    data->P = P;
    data->q = q;
    data->A = A;
    data->l = l;
    data->u = u;

    // Define Solver settings as default
    osqp_set_default_settings(settings);
    settings->warm_start = false;
    settings->polish = this->options_.polish;

    // Setup workspace
    work = osqp_setup(data, settings);

    // Solve Problem
    osqp_solve(work);

    /*
     * SOLUTION
     */
    PolynomialPath solution;
    solution.optimal_cost = work->info->obj_val;
    solution.coefficients.reserve(info.num_dimensions);
    for(size_t dimension_idx = 0; dimension_idx < info.num_dimensions; ++dimension_idx) {
      // Allocate dynamic matrix
      solution.coefficients.emplace_back();
      solution.coefficients[dimension_idx].resize(info.num_params_per_node_per_dim, info.num_nodes);
      solution.coefficients[dimension_idx].fill(0);

      for(size_t node_idx = 0; node_idx < info.num_nodes; ++node_idx) {
        Eigen::MatrixXd coefficients;
        coefficients.resize(info.num_params_per_node_per_dim, 1);
        coefficients.fill(0);
        for(size_t coefficient_idx = 0; coefficient_idx < info.num_params_per_node_per_dim; ++coefficient_idx) {
          const size_t parameter_idx = 0
            // Get to the right dimension
            + info.num_params_per_node_per_dim * info.num_nodes * dimension_idx
            // Get to the right node
            + info.num_params_per_node_per_dim * node_idx
            // Get to the right parameter idx
            + coefficient_idx;
          coefficients(coefficient_idx, 0)
           = work->solution->x[parameter_idx];
        }

        solution.coefficients[dimension_idx].col(node_idx) = coefficients;
      }
    }


    // Cleanup
    osqp_cleanup(work);
    c_free(data->A);
    c_free(data->P);
    c_free(data);
    c_free(settings);

    return solution;
  }

  void PolynomialSolver::Options::Check() {
    if(this->num_dimensions < 1) {
      std::cerr << "PolynomialSolver::Options::Run -- Dimension must be positive." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  
    if(this->polynomial_order < this->derivative_order + 3) {
      std::cerr << "PolynomialSolver::Options::Run -- Polynomial order must be 3 or more orders greater than derivtive order." << std::endl;
      std::exit(EXIT_FAILURE);
    }
  
    if(this->polynomial_order <= this->continuity_order) {
      std::cerr << "PolynomialSolver::Options::Run -- Polynomial order must be greated than continuity order" << std::endl;
      std::exit(EXIT_FAILURE);
    }
  }
}

