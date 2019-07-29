// Author: Tucker Haydon

#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/QR>    

#include "gradient.h"
#include "common.h"

namespace p4 {
  namespace {
    // Upper-level cost function of the form: 
    //   f(x*, y) = 0.5 * x' P x +  c(y)
    //
    // Want to determine the gradient with respect to y, holding x* constant.
    // Although automatic differentiation is not necessary (gradient is just
    // ones()), this class is included for consistency and in case a new
    // time-dependent function is introduced.
    //
    // TODO: Call solver and evaluate
    class ObjectiveCostFunction {
     public:
      ObjectiveCostFunction(const PolynomialSolver::Solution& solver_solution)
        : solver_solution_(solver_solution) {}
    
      static ceres::CostFunction* Create(
          const PolynomialSolver::Solution& solver_solution) {
        auto cost_function = 
          new ceres::DynamicAutoDiffCostFunction<ObjectiveCostFunction>(
            new ObjectiveCostFunction(solver_solution));
        // The number of nodes is equal to the number of times to optimize
        cost_function->AddParameterBlock(solver_solution.constants.num_nodes);
        // There is only one residual: the evaluated cost function
        cost_function->SetNumResiduals(1);

        return cost_function;
      }
    
      template <typename T>
      bool operator()(
          T const* const* times, 
          T* residuals) const {
        // Get the size of the time vector
        const int32_t times_size = this->solver_solution_.constants.num_nodes;

        // Cast constant coefficients to type Jet
        const auto x = 
          Eigen::Map<const Eigen::Matrix<c_float, Eigen::Dynamic, 1>>(
              this->solver_solution_.workspace->solution->x,
              this->solver_solution_.data->n).cast<T>();

        // Cast quadratic matrix to type Jet
        Eigen::SparseMatrix<c_float> P_float;
        OSQP2Eigen(
              this->solver_solution_.data->P,
              P_float);
        const Eigen::SparseMatrix<T> P = P_float.cast<T>();

        // Compose times in Eigen vector
        // times is a vector, but is represented as a double pointer due to the
        // DynamicAutoDiffCostFunction interface
        const auto t =
          Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(times[0], times_size);


        // Fill cost function/residuals
        residuals[0] = (T(0.5) * x.transpose() * P * x + 
          Eigen::Matrix<T, Eigen::Dynamic, 1>::Ones(times_size, 1).transpose() * t).eval()(0,0);
        // residuals[0] = (T(0.5) * x.transpose() * P * x + 
        //   t.transpose() * t).eval()(0,0);
    
        return true;
      }
    
      private:
        PolynomialSolver::Solution solver_solution_;
    };

    // Cost function that encapsulates the cost of constraints of the form: 
    //   g(x*, y) <= 0
    //
    // Want to determine the gradient with respect to y, holding x* constant.
    class ConstraintCostFunction {
     public:
      ConstraintCostFunction(
          const PolynomialSolver::Solution& solver_solution,
          const std::shared_ptr<const PolynomialSolver>& solver)
        : solver_solution_(solver_solution),
          solver_(solver) {}
    
      static ceres::CostFunction* Create(
          const PolynomialSolver::Solution& solver_solution,
          const std::shared_ptr<const PolynomialSolver>& solver) {
        auto cost_function = 
          new ceres::DynamicAutoDiffCostFunction<ConstraintCostFunction>(
            new ConstraintCostFunction(solver_solution, solver));
        // The number of nodes is equal to the number of times to optimize
        cost_function->AddParameterBlock(solver_solution.constants.num_nodes);
        // Multiply by two to account for upper and lower constraints
        cost_function->SetNumResiduals(2*solver_solution.data->m);

        return cost_function;
      }
    
      template <typename T>
      bool operator()(
          T const* const* times, 
          T* residuals) const {
        // Get the size of the time vector
        const int32_t times_size = this->solver_solution_.constants.num_nodes;

        // Cast constant coefficients to type Jet
        const auto x = 
          Eigen::Map<const Eigen::Matrix<c_float, Eigen::Dynamic, 1>>(
              this->solver_solution_.workspace->solution->x,
              this->solver_solution_.data->n).cast<T>();

        // Compose times in std vector
        // times is a vector, but is represented as a double pointer due to the
        // DynamicAutoDiffCostFunction interface
        const std::vector<T> t(times[0], times[0]+times_size);

        // Compose constraints
        Eigen::SparseMatrix<T> A;
        Eigen::Matrix<T, Eigen::Dynamic, 1> l;
        Eigen::Matrix<T, Eigen::Dynamic, 1> u;
        this->solver_->SetConstraints(t, l, u, A);

        // Calculate residuals
        // Residuals are defined as [Ax-u; l-Ax] = 0
        Eigen::Matrix<T, Eigen::Dynamic, 1> upper_residuals = A*x - u;
        Eigen::Matrix<T, Eigen::Dynamic, 1> lower_residuals = l - A*x;

        // Fill residuals
        for(size_t idx = 0; idx < this->solver_solution_.data->m; ++idx) {
          residuals[idx] = upper_residuals(idx);
          residuals[this->solver_solution_.data->m + idx] = lower_residuals(idx);
        }
    
        return true;
      }
    
      private:
        PolynomialSolver::Solution solver_solution_;
        std::shared_ptr<const PolynomialSolver> solver_;
    };

    // RAII buffer for storing a 2D floating point array
    class SmartBuffer2D {
      public:
        SmartBuffer2D(
            const size_t rows, 
            const size_t cols) 
        : rows_(rows),
          cols_(cols) {
          this->data_ = new double*[this->rows_];
          for(size_t row = 0; row < this->rows_; ++row) {
            this->data_[row] = new double[this->cols_];
          }
        }

        ~SmartBuffer2D() {
          for(size_t row = 0; row < this->rows_; ++row) {
            delete[] this->data_[row];
          }
          delete[] this->data_;
        }

        double** Get() const {
          return this->data_;
        }

        size_t Rows() const {
          return this->rows_;
        }

        size_t Cols() const {
          return this->cols_;
        }

      private:
        double** data_;
        size_t rows_;
        size_t cols_;
    };

    // RAII buffer for storing a 1D floating point array
    class SmartBuffer1D {
      public:
        SmartBuffer1D(const size_t rows) 
        : rows_(rows) {
          this->data_ = new double[this->rows_];
        }

        ~SmartBuffer1D() {
          delete[] this->data_;
        }

        double* Get() const {
          return this->data_;
        }

        size_t Rows() const {
          return this->rows_;
        }

      private:
        double* data_;
        size_t rows_;
    };
  }

  PolynomialGradient::Solution PolynomialGradient::Run(
      const std::vector<double>& initial_times,
      const std::shared_ptr<const PolynomialSolver>& solver,
      const PolynomialSolver::Solution& solver_solution) {
    // Prepare initial guess accessor. Need pointer to pointer.
    const double* initial_times_ptr = initial_times.data();

    // Prepare times Eigen vector
    Eigen::Matrix<double, Eigen::Dynamic, 1> times(initial_times.size());
    for(size_t time_idx = 0; time_idx < initial_times.size(); ++time_idx) {
      times(time_idx) = initial_times[time_idx];
    }

    // Extract lagrange multipliers. See documentation in header file for how
    // they are split.
    const auto y = 
      Eigen::Map<const Eigen::Matrix<c_float, Eigen::Dynamic, 1>>(
          solver_solution.workspace->solution->y,
          solver_solution.data->m);
    Eigen::Matrix<double, Eigen::Dynamic, 1> lambda(2*solver_solution.data->m);
    lambda << y.cwiseMax(0), y.cwiseMin(0)*-1;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> objective_jacobian;
    { // Determine the gradient of the objective function
      // Allocate memory
      const size_t num_residuals = 1;
      objective_jacobian.resize(1, num_residuals * solver_solution.constants.num_nodes);

      // Structures for autodiff evaluation
      SmartBuffer1D residuals(num_residuals);
      SmartBuffer2D jacobian(num_residuals, solver_solution.constants.num_nodes);

      // Cost function
      const ceres::CostFunction* cost_function 
        = ObjectiveCostFunction::Create(solver_solution);

      // Evaluate gradient
      bool success = cost_function->Evaluate(
          &initial_times_ptr,
          residuals.Get(),
          jacobian.Get());

      // Notes:
      // http://ceres-solver.org/nnls_modeling.html#_CPPv2N5ceres12CostFunction8EvaluateEPPCdPdPPd
      // jacobians[i][r * parameter_block_sizes_[i] + c] =
      // partial residual[r]
      // -------------------
      // partial parameters[i][c]
      //
      // There is only one parameter block: the time. Thus, the first dimension
      // of the jacobian is always 0.
      if(true == success) {
        for(size_t residual_idx = 0; residual_idx < residuals.Rows(); ++residual_idx) {
          for(size_t parameter_idx = 0; parameter_idx < solver_solution.constants.num_nodes; ++parameter_idx) {
            objective_jacobian(residual_idx, parameter_idx) 
              = jacobian.Get()[0][residual_idx * solver_solution.constants.num_nodes + parameter_idx];
          }
        }
      } else {
        std::cout << "Jacobian Failed!" << std::endl;
      }
    }

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> constraints_jacobian;
    { // Determine the gradient of the constraint function
      // Allocate memory
      const size_t num_residuals = 2*solver_solution.data->m;
      constraints_jacobian.resize(num_residuals, solver_solution.constants.num_nodes);

      // Structures for autodiff evaluation
      SmartBuffer1D residuals(num_residuals);
      SmartBuffer2D jacobian(1, num_residuals * solver_solution.constants.num_nodes);

      // Cost function
      const ceres::CostFunction* cost_function 
        = ConstraintCostFunction::Create(solver_solution, solver);

      // Evaluate gradient
      bool success = cost_function->Evaluate(
          &initial_times_ptr,
          residuals.Get(),
          jacobian.Get());

      // Notes:
      // http://ceres-solver.org/nnls_modeling.html#_CPPv2N5ceres12CostFunction8EvaluateEPPCdPdPPd
      // jacobians[i][r * parameter_block_sizes_[i] + c] =
      // partial residual[r]
      // -------------------
      // partial parameters[i][c]
      //
      // There is only one parameter block: the time. Thus, the first dimension
      // of the jacobian is always 0.
      if(true == success) {
        for(size_t residual_idx = 0; residual_idx < residuals.Rows(); ++residual_idx) {
          for(size_t parameter_idx = 0; parameter_idx < solver_solution.constants.num_nodes; ++parameter_idx) {
            constraints_jacobian(residual_idx, parameter_idx) 
              = jacobian.Get()[0][residual_idx * solver_solution.constants.num_nodes + parameter_idx];
          }
        }
      } else {
        std::cout << "Jacobian Failed!" << std::endl;
      }
    }    
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> gradient = 
      // TODO: Plus or minus lambda??
      (objective_jacobian + lambda.transpose() * constraints_jacobian).transpose();

    { // Project the gradient onto the null space of the timing constraints: Ax-b<=0
      // Resources:
      // http://www2.esm.vt.edu/~zgurdal/COURSES/4084/4084-Docs/LECTURES/GradProj.pdf
      
      // Two inequality constraints are used to constrain the initial node to
      // the specified time. N-1 constaints are used to constrain the time vector
      // to be monotonic, positive
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> A(
          2 + solver_solution.constants.num_nodes - 1, 
          solver_solution.constants.num_nodes);  
      A.fill(0);

      Eigen::Matrix<double, Eigen::Dynamic, 1> b(2 + solver_solution.constants.num_nodes - 1);
      b.fill(0);

      // Constrain initial time 
      A(0,0) = +1;
      A(1,0) = -1;
      b(0) = initial_times[0];
      b(1) = initial_times[0];

      // Constrain time vector to be monotonic, positive
      A.block(2,0,solver_solution.constants.num_nodes - 1, solver_solution.constants.num_nodes-1).diagonal(0) =
        Eigen::Matrix<double, Eigen::Dynamic, 1>::Ones(solver_solution.constants.num_nodes - 1);
      A.block(2,1,solver_solution.constants.num_nodes - 1, solver_solution.constants.num_nodes-1).diagonal(0) =
        -1 * Eigen::Matrix<double, Eigen::Dynamic, 1>::Ones(solver_solution.constants.num_nodes - 1);

      // Select active constraints
      const Eigen::Matrix<double, Eigen::Dynamic, 1> timing_constraints = A * times - b;
      std::vector<size_t> active_constraint_indices;
      for(size_t constraint_idx = 0; constraint_idx < timing_constraints.rows(); ++constraint_idx) {
        // If the constraint is approximately zero, consider it active
        if(abs(timing_constraints(constraint_idx)) < 1e-4) {
          active_constraint_indices.push_back(constraint_idx);
        }
      }

      // N is the matrix of active constraints
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> N(
          active_constraint_indices.size(), 
          solver_solution.constants.num_nodes);
      for(size_t constraint_idx = 0; constraint_idx < N.rows(); ++constraint_idx) {
        N.row(constraint_idx) = A.row(active_constraint_indices[constraint_idx]);
      }

      // Project the gradient onto the null space of the constraints
      // Must use pseudo-inverse as the constraint matrix is often singular. I
      // believe this 
      Eigen::Matrix<double, Eigen::Dynamic, 1> s = 
        (
         Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Identity(
           solver_solution.constants.num_nodes, 
           solver_solution.constants.num_nodes) 
         - N.transpose() * (N * N.transpose()).completeOrthogonalDecomposition().pseudoInverse() * N)
        * gradient;

      gradient = s.normalized();
    }

    // Compose solution
    Solution solution;
    solution.gradient = gradient;

    return solution;
  }
}
